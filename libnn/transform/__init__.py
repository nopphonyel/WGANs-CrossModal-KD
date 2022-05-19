from torch import nn


class TanhRescale(nn.Module):
    """
    Rescale the value tensor to between -1 and 1
    """

    def __init__(self, min, max, padding=0):
        """
        :param min: Min value in the tensor that want to normalize
        :param max: Max value in the tensor
        """
        super(TanhRescale, self).__init__()
        self.mn = min - padding
        self.db2 = ((max + padding) - min) / 2.0

    def forward(self, x):
        return ((x - self.mn) / self.db2) - 1.0