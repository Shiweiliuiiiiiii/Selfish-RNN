import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


class LockedDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m, requires_grad=False) / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x


def embedded_dropout(embed, words, dropout=0.1, scale=None):
    if dropout:
        mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(
            embed.weight) / (1 - dropout)
        masked_embed_weight = mask * embed.weight
    else:
        masked_embed_weight = embed.weight
    if scale:
        masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight

    padding_idx = embed.padding_idx
    if padding_idx is None:
        padding_idx = -1

    X = torch.nn.functional.embedding(words, masked_embed_weight,
                                      padding_idx, embed.max_norm, embed.norm_type,
                                      embed.scale_grad_by_freq, embed.sparse
                                      )
    return X


class HighwayBlock(nn.Module):
    """ Highway Layer Block. Can be used as a highway layer or stacked into a recurrent highway. """

    def __init__(self, in_features, out_features, first=False, couple=False, input_drop=0.5, hidden_drop=0.25):
        super(HighwayBlock, self).__init__()
        self.first = first
        self.couple = couple
        self.out_features = out_features
        if self.first:
            self.W = nn.Linear(in_features, 2*out_features)
            if not couple: self.W_C = nn.Linear(in_features, out_features)
        self.R = nn.Linear(in_features, 2*out_features)
        if not couple: self.R_C = nn.Linear(in_features, out_features)

    def forward(self, x, s, mask):
        if self.first:
            W_H, W_T = torch.split(self.W(x), self.out_features, 1)
            R_H, R_T = torch.split(self.R(s*mask), self.out_features, 1)
            h = torch.tanh(W_H + R_H)
            t = torch.sigmoid(W_T + R_T)
            if self.couple:
                c = 1 - t
            else:
                c = torch.sigmoid(self.W_C(x) + self.R_C(s*mask))
        else:
            R_H, R_T = torch.split(self.R(s*mask), self.out_features, 1)
            h = torch.tanh(R_H)
            t = torch.sigmoid(R_T)
            if self.couple:
                c = 1 - t
            else:
                c = torch.sigmoid(self.R_C(s*mask))
        return h * t + s * c



class RecurrentHighway(nn.Module):
    """Recurrent Highway Layer. Stacks highway blocks with a recurrence mechanism. Replaces LSTM/GRU."""

    def __init__(self, in_features, out_features, recurrence_depth=5, couple=False, input_drop=0.75, hidden_drop=0.25):
        super(RecurrentHighway, self).__init__()
        self.highways = [HighwayBlock(in_features, out_features,
                                      first=True if l == 0 else False,
                                      couple=couple, input_drop=input_drop, hidden_drop=hidden_drop) for l in
                         range(recurrence_depth)]
        self.highways = nn.ModuleList(self.highways)
        self.recurrence_depth = recurrence_depth
        self.hidden_dim = out_features
        self.hidden_drop = hidden_drop

    def rhn_drop_mask(self, x, dropout):
        if not self.training or not dropout:
            return torch.ones_like(x, requires_grad=False)
        else:
            m = x.data.new(x.size(0), x.size(1)).bernoulli_(1 - dropout)
            mask = Variable(m, requires_grad=False) / (1 - dropout)
            return mask

    def forward(self, inp, hidden):
        # expects input dimensions [seq_len, bs, inp_dim]
        outputs = []
        # dropout mask of hidden of all steps
        mask = self.rhn_drop_mask(hidden, self.hidden_drop)
        for x in inp:  # each step
            for block in self.highways:  # depth
                hidden = block(x, hidden, mask)

            outputs.append(hidden)
        outputs = torch.stack(outputs)
        return outputs, hidden

class RHN(nn.Module):
    """Recurrent Highway Networks. Stacks highway blocks with a recurrence mechanism. Replaces LSTM/GRU."""
    def __init__(self, vocab_sz, embedding_dim, hidden_dim, recurrence_depth=1,
                 num_layers=1, input_dp=0.0, output_dp=0.0, hidden_dp=0.0, embed_dp=0.0, tie_weights=True,
                 couple=False):
        super(RHN, self).__init__()
        self.embedding = nn.Embedding(vocab_sz, embedding_dim)
        self.rnns = [RecurrentHighway(embedding_dim if l == 0 else hidden_dim,
                                      (hidden_dim if l != num_layers - 1 else embedding_dim) if tie_weights else hidden_dim,
                                      recurrence_depth=recurrence_depth, couple=couple, input_drop=input_dp,
                                      hidden_drop=hidden_dp) for l in range(num_layers)]
        self.rnns = nn.ModuleList(self.rnns)
        self.fc1 = nn.Linear(embedding_dim if tie_weights else hidden_dim, vocab_sz)

        self.embed_drop = embed_dp
        self.output_drop = output_dp
        self.hidden_dim = hidden_dim
        self.input_dp = input_dp

        self.lockdrop = LockedDropout()

        if tie_weights:
            self.fc1.weight = self.embedding.weight

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc1.bias.data.fill_(0)
        self.fc1.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, bs):
        # Returns a list of zeroed hidden states of dimensions [bs, hidden_dim]
        weight = next(self.parameters()).data
        hidden = weight.new(bs, self.hidden_dim).zero_()

        return hidden

    def forward(self, x, hidden):
        bptt_len, bs = x.shape
        vocab_sz = self.embedding.num_embeddings

        emb = embedded_dropout(self.embedding, x, dropout=self.embed_drop if self.training else 0)
        out = self.lockdrop(emb, self.input_dp)

        for i, rnn in enumerate(self.rnns):
            out, hidden = rnn(out, hidden)
            out = self.lockdrop(out, self.output_drop)

        out = self.fc1(out.flatten(0, 1))
        out = out.view(bptt_len, bs, vocab_sz)
        return out, hidden


class Stacked_LSTM(nn.Module):
    """Stacked LSTM"""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(Stacked_LSTM, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden