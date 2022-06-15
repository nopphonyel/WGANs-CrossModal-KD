import math
import torch
from torch import nn
from torch.nn import functional as F


class Embed(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Embed, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = F.normalize(x, p=2, dim=1)

        return x


class ContrastLoss(nn.Module):
    '''
    contrastive loss, corresponding to Eq.(18)
    '''

    def __init__(self, n_data, eps=1e-7):
        super(ContrastLoss, self).__init__()
        self.n_data = n_data
        self.eps = eps

    def forward(self, x):
        bs = x.size(0)
        N = x.size(1) - 1
        M = float(self.n_data)

        # loss for positive pair
        pos_pair = x.select(1, 0)
        log_pos = torch.div(pos_pair, pos_pair.add(N / M + self.eps)).log_()

        # loss for negative pair
        neg_pair = x.narrow(1, 1, N)
        log_neg = torch.div(neg_pair.clone().fill_(N / M), neg_pair.add(N / M + self.eps)).log_()

        loss = -(log_pos.sum() + log_neg.sum()) / bs

        return loss


class ContrastMemory(nn.Module):
    def __init__(self, feat_dim, n_data, nce_n, nce_t, nce_mom):
        super(ContrastMemory, self).__init__()
        self.N = nce_n
        self.T = nce_t
        self.momentum = nce_mom
        self.Z_t = None
        self.Z_s = None

        stdv = 1. / math.sqrt(feat_dim / 3.)
        self.register_buffer('memory_t', torch.rand(n_data, feat_dim).mul_(2 * stdv).add_(-stdv))
        self.register_buffer('memory_s', torch.rand(n_data, feat_dim).mul_(2 * stdv).add_(-stdv))

    def forward(self, feat_s, feat_t, idx, sample_idx):
        bs = feat_s.size(0)
        feat_dim = self.memory_s.size(1)
        n_data = self.memory_s.size(0)

        # using teacher as anchor
        weight_s = torch.index_select(self.memory_s, 0, sample_idx.view(-1)).detach()
        weight_s = weight_s.view(bs, self.N + 1, feat_dim)
        out_t = torch.bmm(weight_s, feat_t.view(bs, feat_dim, 1))
        out_t = torch.exp(torch.div(out_t, self.T)).squeeze().contiguous()

        # using student as anchor
        weight_t = torch.index_select(self.memory_t, 0, sample_idx.view(-1)).detach()
        weight_t = weight_t.view(bs, self.N + 1, feat_dim)
        out_s = torch.bmm(weight_t, feat_s.view(bs, feat_dim, 1))
        out_s = torch.exp(torch.div(out_s, self.T)).squeeze().contiguous()

        # set Z if haven't been set yet
        if self.Z_t is None:
            self.Z_t = (out_t.mean() * n_data).detach().item()
        if self.Z_s is None:
            self.Z_s = (out_s.mean() * n_data).detach().item()

        out_t = torch.div(out_t, self.Z_t)
        out_s = torch.div(out_s, self.Z_s)

        # update memory
        with torch.no_grad():
            pos_mem_t = torch.index_select(self.memory_t, 0, idx.view(-1))
            pos_mem_t.mul_(self.momentum)
            pos_mem_t.add_(torch.mul(feat_t, 1 - self.momentum))
            pos_mem_t = F.normalize(pos_mem_t, p=2, dim=1)
            self.memory_t.index_copy_(0, idx, pos_mem_t)

            pos_mem_s = torch.index_select(self.memory_s, 0, idx.view(-1))
            pos_mem_s.mul_(self.momentum)
            pos_mem_s.add_(torch.mul(feat_s, 1 - self.momentum))
            pos_mem_s = F.normalize(pos_mem_s, p=2, dim=1)
            self.memory_s.index_copy_(0, idx, pos_mem_s)

        return out_s, out_t
