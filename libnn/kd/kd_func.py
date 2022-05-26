import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftTarget(nn.Module):
    """
    Distilling the Knowledge in a Neural Network
    https://arxiv.org/pdf/1503.02531.pdf
    """

    def __init__(self, temperature):
        """
        :param temperature: Temperature term which used in soft target
        """
        super(SoftTarget, self).__init__()
        self.T = temperature

    def forward(self, out_s, out_t):
        loss = F.kl_div(F.log_softmax(out_s / self.T, dim=1),
                        F.softmax(out_t / self.T, dim=1),
                        reduction='batchmean') * self.T * self.T
        return loss


class Logits(nn.Module):
    '''
    Do Deep Nets Really Need to be Deep?
    http://papers.nips.cc/paper/5484-do-deep-nets-really-need-to-be-deep.pdf
    '''

    def __init__(self):
        super(Logits, self).__init__()

    def forward(self, out_s, out_t):
        loss = F.mse_loss(out_s, out_t)
        return loss


class AT(nn.Module):
    '''
    Paying More Attention to Attention: Improving the Performance of Convolutional
    Neural Netkworks wia Attention Transfer
    https://arxiv.org/pdf/1612.03928.pdf
    '''

    def __init__(self, p):
        super(AT, self).__init__()
        self.p = p

    def forward(self, fm_s, fm_t):
        loss = F.mse_loss(self.attention_map(fm_s), self.attention_map(fm_t))
        return loss

    def attention_map(self, fm, eps=1e-6):
        am = torch.pow(torch.abs(fm), self.p)
        am = torch.sum(am, dim=1, keepdim=True)
        norm = torch.norm(am, dim=(2, 3), keepdim=True)
        am = torch.div(am, norm + eps)
        return am
