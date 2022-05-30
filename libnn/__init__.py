from torch import nn
from libnn import transform

class DeptSepConv2d(nn.Module):
    def __init__(self, nin, nout, kernel_size=3, padding=1, bias=False):
        super(DeptSepConv2d, self).__init__()
        self.dw = nn.Conv2d(nin, nin, kernel_size=kernel_size, padding=padding, groups=nin, bias=bias)
        self.pw = nn.Conv2d(nin, nout, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.dw(x)
        out = self.pw(out)
        return out


class DeptSepConvTranspose2d(nn.Module):
    def __init__(self, nin, nout, kernel_size=3, stride=1, padding=1, bias=False):
        super(DeptSepConvTranspose2d, self).__init__()
        self.dw = nn.ConvTranspose2d(
            in_channels=nin,
            out_channels=nin,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=nin,
            bias=bias)
        self.pw = nn.ConvTranspose2d(nin, nout, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.dw(x)
        out = self.pw(out)
        return out
