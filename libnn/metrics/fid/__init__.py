import torch
import numpy as np

from torch import nn
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d

from libnn.metrics.fid.inception import fid_inception_v3


class FIDOrig(nn.Module):
    def __init__(self, custom_fe: nn.Module = None):
        super(FIDOrig, self).__init__()
        if custom_fe is None:
            self.fe = fid_inception_v3().eval()
        else:
            self.fe = custom_fe.eval()
        self.eps = 1e-6

    def to(self, new_dev):
        self.fe.to(new_dev)
        return self

    def __get_act_stat(self, x_batch):

        with torch.no_grad():
            pred = self.fe(x_batch)

        # if pred.size(2) != 1 or pred.size(3) != 1:
        #     pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
        #
        # pred = pred.squeeze(3).squeeze(2).cpu().numpy()
        pred = pred.cpu().numpy()

        mu = np.mean(pred, axis=0)
        sigma = np.cov(pred, rowvar=False)

        return mu, sigma

    def __call__(self, real_batch, gen_batch):

        mu1, sigma1 = self.__get_act_stat(real_batch)
        mu2, sigma2 = self.__get_act_stat(gen_batch)

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % self.eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * self.eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real
        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1)
                + np.trace(sigma2) - 2 * tr_covmean)


class FID:
    def __init__(self, eps=1e-6):
        self.eps = eps

    def __calc_stat(self, x):
        mu = np.mean(x, axis=0)
        sigma = np.cov(x, rowvar=False)
        return mu, sigma

    def __call__(self, real_batch, gen_batch):
        rb = real_batch.detach().cpu().flatten(start_dim=1).numpy()
        gb = gen_batch.detach().cpu().flatten(start_dim=1).numpy()

        mu1, sigma1 = self.__calc_stat(rb)
        mu2, sigma2 = self.__calc_stat(gb)

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % self.eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * self.eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real
        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1)
                + np.trace(sigma2) - 2 * tr_covmean)


if __name__ == "__main__":
    import torchvision
    import libnn
    from dataset.fMRI_HC import DataLoader
    from dataset.fMRI_HC import fMRI_HC_Dataset

    dev = "cuda"

    # fid = FIDOrig().to(dev)
    fid = FID()
    img_tf = nn.Sequential(
        torchvision.transforms.Resize((64, 64)),
        # libnn.transform.GreyScaleToRGB(),
        libnn.transform.TanhRescale(min=0, max=255, padding=0.01)
    )
    ds = fMRI_HC_Dataset(p_id=1, v=1, img_tf=img_tf).to(dev)
    ld = DataLoader(ds, batch_size=4)

    for i, (fmri, img, label_idx) in enumerate(ld):
        score = fid(real_batch=img, gen_batch=img)
        print(score)
        pass
