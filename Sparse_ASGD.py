import math
import torch
from torch.optim.optimizer import Optimizer


class Sparse_ASGD(Optimizer):
    """Implements Sparse Averaged Stochastic Gradient Descent.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        lambd (float, optional): decay term (default: 1e-4)
        alpha (float, optional): power for eta update (default: 0.75)
        t0 (float, optional): point at which to start averaging (default: 1e6)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """

    def __init__(self, params, lr=1e-2, lambd=1e-4, alpha=0.75, t0=1e6, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, lambd=lambd, alpha=alpha, t0=t0,
                        weight_decay=weight_decay)
        super(Sparse_ASGD, self).__init__(params, defaults)
        self.masks = {}

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for layer, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                if layer in self.masks:
                    grad = p.grad.data
                    if grad.is_sparse:
                        raise RuntimeError('ASGD does not support sparse gradients')
                    state = self.state[p]
                    # State initialization for sparse ASGD
                    if len(state) == 0:
                        state['step'] = torch.zeros_like(p.data)
                        state['eta'] = group['lr']
                        state['mu'] = torch.ones_like(p.data)
                        state['ax'] = torch.zeros_like(p.data)

                    state['step'][self.masks[layer]!=1] = 0
                    state['step'][self.masks[layer]==1] += 1

                    if group['weight_decay'] != 0:
                        grad = grad.add(group['weight_decay'], p.data)

                    # decay term
                    p.data.mul_(1 - group['lambd'] * state['eta'])

                    # update parameter
                    p.data.add_(-state['eta'], grad)

                    # update eta and mu
                    state['mu'][(state['step'] == 0).byte() | (state['step'] == 1).byte() | (state['step'] == 2).byte()] = 1
                    state['mu'][(state['step'] != 0).byte() & (state['step'] != 1).byte() & (state['step'] != 2).byte()] = 1 / (state['step'][(state['step'] != 0).byte() & (state['step'] != 1).byte() & (state['step'] != 2).byte()] - 1)

                    # averaging
                    state['ax'][state['mu'] == 1] = p.data[state['mu'] == 1].clone()
                    state['ax'][state['mu'] != 1] = state['ax'][state['mu'] != 1].add_(p.data[state['mu'] != 1].sub(state['ax'][state['mu'] != 1]).mul(state['mu'][state['mu'] != 1]))

                    # clear non-existing ax
                    state['ax'] = state['ax'] * self.masks[layer]
                else:
                    #dense layer update
                    grad = p.grad.data
                    if grad.is_sparse:
                        raise RuntimeError('ASGD does not support sparse gradients')
                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        state['step'] = 0
                        state['eta'] = group['lr']
                        state['mu'] = 1
                        state['ax'] = torch.zeros_like(p.data)

                    state['step'] += 1

                    if group['weight_decay'] != 0:
                        grad = grad.add(group['weight_decay'], p.data)

                    # decay term
                    p.data.mul_(1 - group['lambd'] * state['eta'])

                    # update parameter
                    p.data.add_(-state['eta'], grad)

                    # averaging
                    if state['mu'] != 1:
                        state['ax'].add_(p.data.sub(state['ax']).mul(state['mu']))
                    else:
                        state['ax'].copy_(p.data)

                    #update eta and mu
                    state['eta'] = (group['lr'] /
                                    math.pow((1 + group['lambd'] * group['lr'] * state['step']), group['alpha']))
                    state['mu'] = 1 / max(1, state['step'] - group['t0'])

        return loss