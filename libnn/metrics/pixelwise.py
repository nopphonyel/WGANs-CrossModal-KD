import torch
import torch.nn.functional as nnf


class PixelWise:
    def __init__(self):
        pass

    def __call__(self, real_batch: torch.Tensor, gen_batch: torch.Tensor):
        return nnf.mse_loss(real_batch, gen_batch)


class MaskedPixelWise:
    def __init__(self):
        pass

    def __call__(
            self,
            real_batch: torch.Tensor,
            real_label: torch.Tensor,
            gen_batch: torch.Tensor,
            fy_label: torch.Tensor
    ):
        m = real_label.clone().detach() == fy_label.clone().detach()
        s = torch.sum(m)
        l = nnf.mse_loss(real_batch, gen_batch, reduction='none')
        loss = torch.mean(l.flatten(1) * m.unsqueeze(1), dim=1)
        return torch.sum(loss)/s
