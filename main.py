import argparse
import time
import math
import os
import torch
import sys
import data
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import sparse_rnn_core
from sparse_rnn_core import Masking, CosineDecay
from models import RHN, Stacked_LSTM
from Sparse_ASGD import Sparse_ASGD

parser = argparse.ArgumentParser(description='PyTorch implementation of Selfish-RNN')
parser.add_argument('--data', type=str, default='data/penn/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RHN, LSTM)')
parser.add_argument('--evaluate', default='', type=str, metavar='PATH',
                       help='path to pre-trained model (default: none)')
parser.add_argument('--emsize', type=int, default=1500,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1500,
                    help='number of hidden units per layer')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--nrecurrence_depth', type=int, default=10,
                    help='number of recurrence layer')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.0)')
parser.add_argument('--beta', type=float, default=1,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--finetuning', type=int, default=100,
                    help='When (which epochs) to switch to finetuning')
parser.add_argument('--lr', type=float, default=15,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--eval_batch_size', type=int, default=20,
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.65,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.25,
                    help='dropout for rnn hidden units (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.65,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.2,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--couple', action='store_true',
                    help='couple the transform and carry weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--optimizer', type=str, default='sgd',
                    help='optimizer to use (sgd, adam)')
parser.add_argument('--testmodel', type=str, default='15877701169307067.pt',
                    help='the name of saved model')
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--save', type=str, default=randomhash + '.pt',
                    help='path to save the final model')
sparse_rnn_core.add_sparse_args(parser)
args = parser.parse_args()

print(args)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

def model_save(fn):
    torch.save(model.state_dict(), fn)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)
###############################################################################
# Load data
###############################################################################
corpus = data.Corpus(args.data)
# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, args.eval_batch_size)
test_data = batchify(corpus.test, args.eval_batch_size)
###############################################################################
# Build the model
###############################################################################
ntokens = len(corpus.dictionary)

if args.model == 'RHN':
    model = RHN(vocab_sz=ntokens, embedding_dim=args.emsize, hidden_dim=args.nhid,
                             recurrence_depth=args.nrecurrence_depth, num_layers=args.nlayers, input_dp=args.dropouti,
                             output_dp=args.dropout, hidden_dp=args.dropouth, embed_dp=args.dropoute,
                             tie_weights=args.tied, couple=args.couple).to(device)
elif args.model == 'LSTM':
    model = Stacked_LSTM(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(device)

criterion = nn.CrossEntropyLoss()
###############################################################################
# Train and evaluate code
###############################################################################
def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i)

            output, hidden = model(data, hidden)
            hidden = repackage_hidden(hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)

def train(mask=None):
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)

    if args.model != 'Transformer':
        hidden = model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        optimizer.zero_grad()
        hidden = repackage_hidden(hidden)
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        if mask is not None:
            mask.step()
        else:
            optimizer.step()
        total_loss += loss.item()
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
            sys.stdout.flush()
###############################################################################
# Training
###############################################################################
if args.evaluate:
    print("=> loading checkpoint '{}'".format(args.evaluate))
    model.load_state_dict(torch.load(args.evaluate))
    print('=> testing...')
    test_loss = evaluate(test_data)
    print('=' * 89)
    print('| Final test | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)
    sys.stdout.flush()
else:
    mask = None
    lr = args.lr
    best_val_loss = []
    stored_loss = 100000000
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        optimizer = None
        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
        if args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), betas=(0, 0.999), eps=1e-9, weight_decay=args.wdecay)
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.5, patience=2, threshold=0)

        mask = None
        if args.sparse:
            decay = CosineDecay(args.death_rate, args.epochs * len(train_data) // args.bptt)
            mask = Masking(optimizer, death_rate=args.death_rate, death_mode=args.death, death_rate_decay=decay, growth_mode=args.growth,
                           redistribution_mode=args.redistribution, model=args.model)
            mask.add_module(model, sparse_init=args.sparse_init, density=args.density)

        # Loop over epochs.
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            train(mask)
            if 't0' in optimizer.param_groups[0]:
                tmp = {}
                for prm in model.parameters():
                    tmp[prm] = prm.data.clone()
                    prm.data = optimizer.state[prm]['ax'].clone()
                val_loss2 = evaluate(val_data)
                print('-' * 89)
                print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                      'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                    epoch, (time.time() - epoch_start_time), val_loss2, math.exp(val_loss2), val_loss2 / math.log(2)))
                print('-' * 89)

                if val_loss2 < stored_loss:
                    model_save(args.save)
                    print('Saving Averaged!')
                    stored_loss = val_loss2

                for prm in model.parameters():
                    prm.data = tmp[prm].clone()

                if args.sparse and epoch < args.epochs + 1:
                    mask.at_end_of_epoch(epoch)

            else:
                val_loss = evaluate(val_data)
                print('-' * 89)
                print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                      'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                    epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss), val_loss / math.log(2)))
                print('-' * 89)

                if val_loss < stored_loss:
                    model_save(args.save)
                    print('Saving model (new best validation)')
                    stored_loss = val_loss

                if args.optimizer == 'adam':
                    scheduler.step(val_loss)

                if args.optimizer == 'sgd' and 't0' not in optimizer.param_groups[0] and (
                        len(best_val_loss) > args.nonmono and val_loss > min(best_val_loss[:-args.nonmono])):
                    print('Switching to ASGD')
                    optimizer = Sparse_ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
                    mask.optimizer = optimizer
                    mask.init_optimizer_mask()

                if args.sparse and 't0' not in optimizer.param_groups[0]:
                    mask.at_end_of_epoch(epoch)

                best_val_loss.append(val_loss)

            print("PROGRESS: {}%".format((epoch / args.epochs) * 100))

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    with open(args.save, 'rb') as f:
        model.load_state_dict(torch.load(args.save))

    # Run on test data.
    test_loss = evaluate(test_data)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)
    sys.stdout.flush()
