from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math

def add_sparse_args(parser):
    parser.add_argument('--growth', type=str, default='random', help='Growth mode. Choose from: momentum, random, gradient.')
    parser.add_argument('--death', type=str, default='magnitude', help='Death mode / pruning mode. Choose from: magnitude, SET, threshold.')
    parser.add_argument('--redistribution', type=str, default='none', help='Redistribution mode. Choose from: momentum, magnitude, nonzeros, or none.')
    parser.add_argument('--death-rate', type=float, default=0.50, help='The pruning rate / death rate.')
    parser.add_argument('--density', type=float, default=0.33, help='The density of the overall sparse network.')
    parser.add_argument('--sparse', action='store_true', help='Enable sparse mode. Default: True.')

class CosineDecay(object):
    def __init__(self, death_rate, T_max, eta_min=0.005, last_epoch=-1):
        self.sgd = optim.SGD(torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1))]), lr=death_rate)
        self.cosine_stepper = torch.optim.lr_scheduler.CosineAnnealingLR(self.sgd, T_max, eta_min, last_epoch)

    def step(self):
        self.cosine_stepper.step()

    def get_dr(self, death_rate):
        return self.sgd.param_groups[0]['lr']

class LinearDecay(object):
    def __init__(self, death_rate, factor=0.99, frequency=600):
        self.factor = factor
        self.steps = 0
        self.frequency = frequency

    def step(self):
        self.steps += 1

    def get_dr(self, death_rate):
        if self.steps > 0 and self.steps % self.frequency == 0:
            return death_rate*self.factor
        else:
            return death_rate


class Masking(object):
    def __init__(self, optimizer, death_rate=0.3, growth_death_ratio=1.0, death_rate_decay=None, death_mode='magnitude', growth_mode='momentum', redistribution_mode='momentum', threshold=0.001, model='LSTM'):
        growth_modes = ['random', 'momentum', 'momentum_neuron']
        if growth_mode not in growth_modes:
            print('Growth mode: {0} not supported!'.format(growth_mode))
            print('Supported modes are:', str(growth_modes))

        self.growth_mode = growth_mode
        self.death_mode = death_mode
        self.growth_death_ratio = growth_death_ratio
        self.redistribution_mode = redistribution_mode
        self.death_rate_decay = death_rate_decay
        self.model = model

        self.masks = {}
        self.pruning_rate = {}
        self.modules = []
        self.names = []
        self.optimizer = optimizer

        self.adjusted_growth = 0
        self.adjustments = []
        self.baseline_nonzero = None
        self.name2baseline_nonzero = {}

        # stats
        self.name2variance = {}
        self.name2zeros = {}
        self.name2nonzeros = {}
        self.total_variance = 0
        self.total_removed = 0
        self.total_zero = 0
        self.total_nonzero = 0
        self.death_rate = death_rate
        self.name2death_rate = {}
        self.steps = 0

        # lstm stats
        if self.model == 'LSTM':
            self.gate_num = 4
        elif self.model == 'RHN':
            self.gate_num = 2
        self.gates_mask = {}
        self.gates_weight_grad = {}
        self.gates_nonzeros = {}
        self.gates_zeros = {}
        self.gates_weight = {}
        self.gate2variance = {}

        # global growth/death state
        self.threshold = threshold
        self.growth_threshold = threshold
        self.growth_increment = 0.2
        self.increment = 0.2
        self.tolerance = 0.02
        self.prune_every_k_steps = None

    def init(self, mode='enforce_density_per_layer', density=0.05):
        self.sparsity = density
        if mode == 'enforce_density_per_layer':
            self.baseline_nonzero = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    self.masks[name][:] = (torch.rand(weight.shape) < density).float().data.cuda()
                    self.baseline_nonzero += weight.numel()*density
            self.apply_mask()

        elif mode == 'size_proportional':
            # initialization used in sparse evolutionary training
            total_params = 0
            self.baseline_nonzero = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    total_params += weight.numel()
                    self.baseline_nonzero += weight.numel()*density

            target_params = total_params *density
            tolerance = 5
            current_params = 0
            new_nonzeros = 0
            epsilon = 10.0
            growth_factor = 0.5
            # searching for the right epsilon for a specific sparsity level
            while not ((current_params+tolerance > target_params) and (current_params-tolerance < target_params)):
                new_nonzeros = 0.0
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    # original SET formulation for fully connected weights: num_weights = epsilon * (noRows + noCols)
                    # we adapt the same formula for convolutional weights
                    growth =  epsilon*sum(weight.shape)
                    new_nonzeros += growth
                current_params = new_nonzeros
                if current_params > target_params:
                    epsilon *= 1.0 - growth_factor
                else:
                    epsilon *= 1.0 + growth_factor
                growth_factor *= 0.95

            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                growth =  epsilon*sum(weight.shape)
                prob = growth/np.prod(weight.shape)
                self.masks[name][:] = (torch.rand(weight.shape) < prob).float().data.cuda()
            self.apply_mask()
        self.init_death_rate(self.death_rate)
        self.print_nonzero_counts()
        if 't0' in self.optimizer.param_groups[0]:
            # initialize masks for SparseASGD
            self.init_optimizer_mask()

    def init_death_rate(self, death_rate):
        for name in self.masks:
            self.name2death_rate[name] = death_rate

    def init_optimizer_mask(self):
        if self.model == 'LSTM':
            self.optimizer.masks[0] = self.masks['encoder.weight'].clone()
            self.optimizer.masks[1] = self.masks['rnn.weight_ih_l0'].clone()
            self.optimizer.masks[2] = self.masks['rnn.weight_hh_l0'].clone()
            self.optimizer.masks[5] = self.masks['rnn.weight_ih_l1'].clone()
            self.optimizer.masks[6] = self.masks['rnn.weight_hh_l1'].clone()
            self.optimizer.masks[9] = self.masks['decoder.weight'].clone()
        elif self.model == 'RHN':
            self.optimizer.masks[0] = self.masks['embedding.weight'].clone()
            self.optimizer.masks[1] = self.masks['rnns.0.highways.0.W.weight'].clone()
            self.optimizer.masks[3] = self.masks['rnns.0.highways.0.R.weight'].clone()
            self.optimizer.masks[5] = self.masks['rnns.0.highways.1.R.weight'].clone()
            self.optimizer.masks[7] = self.masks['rnns.0.highways.2.R.weight'].clone()
            self.optimizer.masks[9] = self.masks['rnns.0.highways.3.R.weight'].clone()
            self.optimizer.masks[11] = self.masks['rnns.0.highways.4.R.weight'].clone()
            self.optimizer.masks[13] = self.masks['rnns.0.highways.5.R.weight'].clone()
            self.optimizer.masks[15] = self.masks['rnns.0.highways.6.R.weight'].clone()
            self.optimizer.masks[17] = self.masks['rnns.0.highways.7.R.weight'].clone()
            self.optimizer.masks[19] = self.masks['rnns.0.highways.8.R.weight'].clone()
            self.optimizer.masks[21] = self.masks['rnns.0.highways.9.R.weight'].clone()
        elif self.model == 'ONLSTM':
            self.optimizer.masks[0] = self.masks['encoder.weight'].clone()
            self.optimizer.masks[1] = self.masks['rnn.cells.0.ih.0.weight'].clone()
            self.optimizer.masks[5] = self.masks['rnn.cells.1.ih.0.weight'].clone()
            self.optimizer.masks[9] = self.masks['rnn.cells.2.ih.0.weight'].clone()

    def at_end_of_epoch(self,epoch):
        self.truncate_weights(epoch)
        if 't0' in self.optimizer.param_groups[0]:
            self.init_optimizer_mask()
        self.print_nonzero_counts()

    def step(self):
        self.optimizer.step()
        self.apply_mask()
        self.death_rate_decay.step()
        for name in self.masks:
            self.name2death_rate[name] = self.death_rate_decay.get_dr(self.name2death_rate[name])

        self.steps += 1

        if self.prune_every_k_steps is not None:
            if self.steps % self.prune_every_k_steps == 0:
                self.truncate_weights()
                self.print_nonzero_counts()

    def add_module(self, module, density, sparse_init='enforce_density_per_layer'):
        self.modules.append(module)
        for name, tensor in module.named_parameters():
            self.names.append(name)
            self.masks[name] = torch.zeros_like(tensor, dtype=torch.float32, requires_grad=False).cuda()
        self.remove_weight_partial_name('bias')
        self.remove_type(nn.BatchNorm2d)
        self.remove_type(nn.BatchNorm1d)
        self.remove_type(nn.PReLU)
        self.init(mode=sparse_init, density=density)

    def remove_weight(self, name):
        if name in self.masks:
            print('Removing {0} of size {1} = {2} parameters.'.format(name, self.masks[name].shape, self.masks[name].numel()))
            self.masks.pop(name)

    def remove_weight_partial_name(self, partial_name, verbose=False):
        removed = set()
        for name in list(self.masks.keys()):
            if partial_name in name:
                if verbose:
                    print('Removing {0}...'.format(name))
                removed.add(name)
                self.masks.pop(name)
        print('Removed {0} layers.'.format(len(removed)))

        i = 0
        while i < len(self.names):
            name = self.names[i]
            if name in removed: self.names.pop(i)
            else: i += 1


    def remove_type(self, nn_type, verbose=False):
        for module in self.modules:
            for name, module in module.named_modules():
                if isinstance(module, nn_type):
                    self.remove_weight_partial_name(name, verbose=True)

    def apply_mask(self):
        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name in self.masks:
                    tensor.data = tensor.data*self.masks[name]
                    if 'momentum_buffer' in self.optimizer.state[tensor]:
                        self.optimizer.state[tensor]['momentum_buffer'] = self.optimizer.state[tensor]['momentum_buffer']*self.masks[name]

    def truncate_weights(self, epoch):
        self.gather_statistics()
        name2regrowth = self.calc_growth_redistribution()

        total_nonzero_new = 0
        total_removed = 0
        if self.death_mode == 'global_magnitude':
            total_removed = self.global_magnitude_death()
        else:
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    mask = self.masks[name]

                    # death
                    if self.death_mode == 'magnitude':
                        new_mask = self.magnitude_death(mask, weight, name)
                    elif self.death_mode == 'SET':
                        new_mask = self.magnitude_and_negativity_death(mask, weight, name)
                    elif self.death_mode == 'threshold':
                        new_mask = self.threshold_death(mask, weight, name)

                    total_removed += self.name2nonzeros[name] - new_mask.sum().item()
                    self.pruning_rate[name] = (self.name2nonzeros[name] - new_mask.sum().item()) / self.name2nonzeros[name]
                    print("Name:", name, "pruning_rate", self.pruning_rate[name])
                    self.masks[name][:] = new_mask

        if self.growth_mode == 'global_momentum':
            total_nonzero_new = self.global_momentum_growth(total_removed + self.adjusted_growth)
        else:
            if self.death_mode == 'threshold':
                expected_killed = sum(name2regrowth.values())
                if total_removed < (1.0-self.tolerance)*expected_killed:
                    self.threshold *= 2.0
                elif total_removed > (1.0+self.tolerance) * expected_killed:
                    self.threshold *= 0.5

            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    new_mask = self.masks[name].data.byte()

                    if self.death_mode == 'threshold':
                        total_regrowth = math.floor((total_removed/float(expected_killed))*name2regrowth[name]*self.growth_death_ratio)
                    elif self.redistribution_mode == 'none':
                        if name not in self.name2baseline_nonzero:
                            self.name2baseline_nonzero[name] = self.name2nonzeros[name]
                        old = self.name2baseline_nonzero[name]
                        new = new_mask.sum().item()
                        total_regrowth = int(old-new)
                    elif self.death_mode == 'global_magnitude':
                        expected_removed = self.baseline_nonzero*self.name2death_rate[name]
                        expected_vs_actual = total_removed/expected_removed
                        total_regrowth = math.floor(expected_vs_actual*name2regrowth[name]*self.growth_death_ratio)
                    else:
                        total_regrowth = math.floor(name2regrowth[name]*self.growth_death_ratio)

                    # growth
                    if self.growth_mode == 'random_rnn':
                        if 'rnn' in name:
                            new_mask = self.gates_growth(name, new_mask, total_regrowth, weight, epoch)
                        else:
                            new_mask = self.random_growth(name, new_mask, total_regrowth, weight)
                    if self.growth_mode == 'random':
                        new_mask = self.random_growth(name, new_mask, total_regrowth, weight)
                    elif self.growth_mode == 'momentum':
                        new_mask = self.momentum_growth(name, new_mask, total_regrowth, weight)
                    elif self.growth_mode == 'gradient':
                        # implementation for Rigging Ticket
                        new_mask = self.gradient_growth(name, new_mask, total_regrowth, weight)
                    elif self.growth_mode == 'momentum_neuron':
                        new_mask = self.momentum_neuron_growth(name, new_mask, total_regrowth, weight)

                    new_nonzero = new_mask.sum().item()

                    # exchanging masks
                    self.masks.pop(name)
                    self.masks[name] = new_mask.float()
                    total_nonzero_new += new_nonzero
        self.apply_mask()
        # Some growth techniques and redistribution are probablistic and we might not grow enough weights or too much weights
        # Here we run an exponential smoothing over (death-growth) residuals to adjust future growth
        self.adjustments.append(self.baseline_nonzero - total_nonzero_new)
        self.adjusted_growth = 0.25*self.adjusted_growth + (0.75*(self.baseline_nonzero - total_nonzero_new)) + np.mean(self.adjustments)
        print(self.total_nonzero, self.baseline_nonzero, self.adjusted_growth)

        if self.total_nonzero > 0:
            print('old, new nonzero count:', self.total_nonzero, total_nonzero_new, self.adjusted_growth)

    '''
                    REDISTRIBUTION
    '''

    def gather_statistics(self):
        self.name2nonzeros = {}
        self.name2zeros = {}
        self.name2variance = {}

        self.gates_mask = {}
        self.gates_nonzeros = {}
        self.gates_zeros = {}
        self.gates_weight = {}
        self.gate2variance = {}

        self.total_variance = 0.0
        self.total_removed = 0
        self.total_nonzero = 0
        self.total_zero = 0.0
        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name not in self.masks: continue
                mask = self.masks[name]
                if 'rnn' in name:
                    self.redistribution_rnn(name, mask, tensor)
                if self.redistribution_mode == 'momentum':
                    grad = self.get_momentum_for_weight(tensor)
                    self.name2variance[name] = torch.abs(grad[mask.byte()]).mean().item()#/(V1val*V2val)
                elif self.redistribution_mode == 'magnitude':
                    self.name2variance[name] = torch.abs(tensor)[mask.byte()].mean().item()
                elif self.redistribution_mode == 'nonzeros':
                    self.name2variance[name] = float((torch.abs(tensor) > self.threshold).sum().item())
                elif self.redistribution_mode == 'none':
                    self.name2variance[name] = 1.0
                elif self.redistribution_mode == 'uniform_distribution':
                    self.name2variance[name] = 1
                else:
                    print('Unknown redistribution mode:{0}'.format(self.redistribution_mode))
                    raise Exception('Unknown redistribution mode!')

                if not np.isnan(self.name2variance[name]):
                    self.total_variance += self.name2variance[name]
                self.name2nonzeros[name] = mask.sum().item()
                self.name2zeros[name] = mask.numel() - self.name2nonzeros[name]

                sparsity = self.name2zeros[name]/float(self.masks[name].numel())
                death_rate = self.name2death_rate[name]
                if sparsity < 0.2:
                    expected_variance = 1.0/len(list(self.name2variance.keys()))
                    actual_variance = self.name2variance[name]
                    expected_vs_actual = expected_variance/actual_variance
                    if expected_vs_actual < 1.0:
                        death_rate = min(sparsity, death_rate)
                num_remove = math.ceil(death_rate*self.name2nonzeros[name])
                self.total_removed += num_remove
                self.total_nonzero += self.name2nonzeros[name]
                self.total_zero += self.name2zeros[name]

    def calc_growth_redistribution(self):
        num_overgrowth = 0
        total_overgrowth = 0
        residual = 0
        for name in self.name2variance:
            self.name2variance[name] /= self.total_variance
        for name in self.gate2variance:
            self.gate2variance[name] = [(float(i)/sum(self.gate2variance[name])) for i in self.gate2variance[name]]

        residual = 9999
        mean_residual = 0
        name2regrowth = {}
        i = 0
        expected_var = 1.0/len(self.name2variance)
        while residual > 0 and i < 1000:
            residual = 0
            for name in self.name2variance:
                sparsity = self.name2zeros[name]/float(self.masks[name].numel())
                death_rate = self.name2death_rate[name]
                if sparsity < 0.2:
                    expected_variance = 1.0/len(list(self.name2variance.keys()))
                    actual_variance = self.name2variance[name]
                    expected_vs_actual = expected_variance/actual_variance
                    if expected_vs_actual < 1.0:
                        death_rate = min(sparsity, death_rate)
                num_remove = math.ceil(death_rate*self.name2nonzeros[name])
                num_nonzero = self.name2nonzeros[name]
                num_zero = self.name2zeros[name]
                max_regrowth = num_zero + num_remove

                if name in name2regrowth:
                    regrowth = name2regrowth[name]
                else:
                    regrowth = math.ceil(self.name2variance[name]*(self.total_removed+self.adjusted_growth))
                regrowth += mean_residual

                if regrowth > 0.99*max_regrowth:
                    name2regrowth[name] = 0.99*max_regrowth
                    residual += regrowth - name2regrowth[name]
                else:
                    name2regrowth[name] = regrowth
            if len(name2regrowth) == 0: mean_residual = 0
            else:
                mean_residual = residual / len(name2regrowth)
            i += 1

        if i == 1000:
            print('Error resolving the residual! Layers are too full! Residual left over: {0}'.format(residual))

        return name2regrowth


    '''
                    DEATH
    '''
    def Max_MI(self, weight, mask, name):
        new_mask = mask.clone()
        print((new_mask==0).sum())
        pruning_number = self.name2nonzeros[name] * self.name2death_rate[name]
        weights_change = torch.abs(weight.data.view(-1)) - torch.abs(self.pre_tensor[name].data.view(-1))
        num_MD = (weights_change < 0).sum()
        if num_MD < pruning_number:
            pruning_number = num_MD
        pruning_number = int(pruning_number)
        x, idx = torch.sort(weights_change)
        new_mask.data.view(-1)[idx[:pruning_number]] = 0
        print((new_mask == 0).sum())
        return new_mask

    def threshold_death(self, mask, weight, name):
        return (torch.abs(weight.data) > self.threshold)

    def magnitude_death(self, mask, weight, name):
        sparsity = self.name2zeros[name]/float(self.masks[name].numel())
        death_rate = self.name2death_rate[name]
        if sparsity < 0.2:
            expected_variance = 1.0/len(list(self.name2variance.keys()))
            actual_variance = self.name2variance[name]
            expected_vs_actual = expected_variance/actual_variance
            if expected_vs_actual < 1.0:
                death_rate = min(sparsity, death_rate)
                print(name, expected_variance, actual_variance, expected_vs_actual, death_rate)
        num_remove = math.ceil(death_rate*self.name2nonzeros[name])
        if num_remove == 0.0: return weight.data != 0.0
        num_zeros = self.name2zeros[name]

        x, idx = torch.sort(torch.abs(weight.data.view(-1)))
        n = idx.shape[0]
        num_nonzero = n-num_zeros

        k = math.ceil(num_zeros + num_remove)
        threshold = x[k-1].item()

        return (torch.abs(weight.data) > threshold)

    def global_magnitude_death(self):
        death_rate = 0.0
        for name in self.name2death_rate:
            if name in self.masks:
                death_rate = self.name2death_rate[name]
        tokill = math.ceil(death_rate*self.baseline_nonzero)
        total_removed = 0
        prev_removed = 0
        while total_removed < tokill*(1.0-self.tolerance) or (total_removed > tokill*(1.0+self.tolerance)):
            total_removed = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    remain = (torch.abs(weight.data) > self.threshold).sum().item()
                    total_removed += self.name2nonzeros[name] - remain

            if prev_removed == total_removed: break
            prev_removed = total_removed
            if total_removed > tokill*(1.0+self.tolerance):
                self.threshold *= 1.0-self.increment
                self.increment *= 0.99
            elif total_removed < tokill*(1.0-self.tolerance):
                self.threshold *= 1.0+self.increment
                self.increment *= 0.99

        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                self.masks[name][:] = torch.abs(weight.data) > self.threshold

        return int(total_removed)


    def global_momentum_growth(self, total_regrowth):
        togrow = total_regrowth
        total_grown = 0
        last_grown = 0
        while total_grown < togrow*(1.0-self.tolerance) or (total_grown > togrow*(1.0+self.tolerance)):
            total_grown = 0
            total_possible = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue

                    new_mask = self.masks[name]
                    grad = self.get_momentum_for_weight(weight)
                    grad = grad*(new_mask==0).float()
                    possible = (grad !=0.0).sum().item()
                    total_possible += possible
                    grown = (torch.abs(grad.data) > self.growth_threshold).sum().item()
                    total_grown += grown
            print(total_grown, self.growth_threshold, togrow, self.growth_increment, total_possible)
            if total_grown == last_grown: break
            last_grown = total_grown


            if total_grown > togrow*(1.0+self.tolerance):
                self.growth_threshold *= 1.02
            elif total_grown < togrow*(1.0-self.tolerance):
                self.growth_threshold *= 0.98

        total_new_nonzeros = 0
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue

                new_mask = self.masks[name]
                grad = self.get_momentum_for_weight(weight)
                grad = grad*(new_mask==0).float()
                self.masks[name][:] = (new_mask.byte() | (torch.abs(grad.data) > self.growth_threshold)).float()
                total_new_nonzeros += new_mask.sum().item()
        return total_new_nonzeros


    def magnitude_and_negativity_death(self, mask, weight, name):
        num_remove = math.ceil(self.name2death_rate[name]*self.name2nonzeros[name])
        num_zeros = self.name2zeros[name]

        # find magnitude threshold
        # remove all weights which absolute value is smaller than threshold
        x, idx = torch.sort(weight[weight > 0.0].data.view(-1))
        k = math.ceil(num_remove/2.0)
        if k >= x.shape[0]:
            k = x.shape[0]

        threshold_magnitude = x[k-1].item()

        # find negativity threshold
        # remove all weights which are smaller than threshold
        x, idx = torch.sort(weight[weight < 0.0].view(-1))
        k = math.ceil(num_remove/2.0)
        if k >= x.shape[0]:
            k = x.shape[0]
        threshold_negativity = x[k-1].item()
        pos_mask = (weight.data > threshold_magnitude) & (weight.data > 0.0)
        neg_mask = (weight.data > threshold_negativity) & (weight.data < 0.0)
        new_mask = pos_mask | neg_mask
        return new_mask

    '''
                    GROWTH
    '''
    def random_growth(self, name, new_mask, total_regrowth, weight):
        n = (new_mask==0).sum().item()
        if n == 0: return new_mask
        expeced_growth_probability = (total_regrowth/n)
        new_weights = torch.rand(new_mask.shape).cuda() < expeced_growth_probability
        new_mask_final = new_mask.byte() | new_weights
        return new_mask_final

    def momentum_growth(self, name, new_mask, total_regrowth, weight):
        grad = self.get_momentum_for_weight(weight)
        grad = grad*(new_mask==0).float()
        y, idx = torch.sort(torch.abs(grad).flatten(), descending=True)
        new_mask.data.view(-1)[idx[:total_regrowth]] = 1.0

        return new_mask

    def gradient_growth(self, name, new_mask, total_regrowth, weight):
        print('implement gradient regrow:')
        grad = self.get_gradient_for_weights(weight)
        grad = grad*(new_mask==0).float()
        y, idx = torch.sort(torch.abs(grad).flatten(), descending=True)
        new_mask.data.view(-1)[idx[:total_regrowth]] = 1.0

        return new_mask
    def momentum_neuron_growth(self, name, new_mask, total_regrowth, weight):
        grad = self.get_momentum_for_weight(weight)

        M = torch.abs(grad)
        if len(M.shape) == 2: sum_dim = [1]
        elif len(M.shape) == 4: sum_dim = [1, 2, 3]

        v = M.mean(sum_dim).data
        v /= v.sum()

        slots_per_neuron = (new_mask==0).sum(sum_dim)

        M = M*(new_mask==0).float()
        for i, fraction  in enumerate(v):
            neuron_regrowth = math.floor(fraction.item()*total_regrowth)
            available = slots_per_neuron[i].item()

            y, idx = torch.sort(M[i].flatten())
            if neuron_regrowth > available:
                neuron_regrowth = available
            threshold = y[-(neuron_regrowth)].item()
            if threshold == 0.0: continue
            if neuron_regrowth < 10: continue
            new_mask[i] = new_mask[i] | (M[i] > threshold)

        return new_mask

    '''
                UTILITY
    '''
    def redistribution_rnn(self, name, mask, weight):
        self.gates_mask[name] = torch.chunk(mask, self.gate_num, 0)
        self.gates_weight[name] = torch.chunk(weight, self.gate_num, 0)
        self.gates_weight_grad[name] = torch.chunk(weight.grad, self.gate_num, 0)
        self.gates_nonzeros[name] = [self.gates_mask[name][i].sum().item() for i in range(self.gate_num)]
        self.gates_zeros[name] = [self.gates_mask[name][i].numel() - self.gates_nonzeros[name][i] for i in
                                  range(self.gate_num)]

        self.gate2variance[name] = [1/self.gate_num for i in range(self.gate_num)]


    def get_momentum_for_weight(self, weight):
        if 'exp_avg' in self.optimizer.state[weight]:
            adam_m1 = self.optimizer.state[weight]['exp_avg']
            adam_m2 = self.optimizer.state[weight]['exp_avg_sq']
            grad = adam_m1/(torch.sqrt(adam_m2) + 1e-08)
        elif 'momentum_buffer' in self.optimizer.state[weight]:
            grad = self.optimizer.state[weight]['momentum_buffer']
        return grad

    def get_gradient_for_weights(self, weight):
        grad = weight.grad.clone()
        return grad

    def print_nonzero_counts(self):
        gate_name = ['i', 'f', 'c', 'o']
        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name not in self.masks: continue
                mask = self.masks[name]
                num_nonzeros = (mask != 0).sum().item()
                if name in self.name2variance:
                    val = '{0}: {1}->{2}, density: {3:.3f}, proportion: {4:.4f}'.format(name, self.name2nonzeros[name], num_nonzeros, num_nonzeros/float(mask.numel()), self.name2variance[name])
                    print(val)
                    if 'rnn' in name:
                        newmasks_print = torch.chunk(mask, self.gate_num, 0)
                        num_newmasks_print = [(newmasks_print[i] != 0).sum().item() for i in range(self.gate_num)]
                        for i in range(len(num_newmasks_print)):
                            val_gate = '{0}: {1}->{2}, density: {3:.3f}, proportion: {4:.4f}'.format(gate_name[i], self.gates_nonzeros[name][i], num_newmasks_print[i],num_newmasks_print[i] / float(newmasks_print[i].numel()), self.gate2variance[name][i])
                            print(val_gate)
                else:
                    print(name, num_nonzeros)

        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name not in self.masks: continue
                print('Death rate: {0}\n'.format(self.name2death_rate[name]))
                break
