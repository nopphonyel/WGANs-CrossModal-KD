import torch
from torch import nn


# Generator Code
class Generator(nn.Module):
    def __init__(self, ngpu, num_classes, embed_size=100, z_dim=100, latent_size=200, img_channel=3, gen_dim=64):
        """
        :param ngpu:
        :param num_classes: number of image class (use for do label embedding)
        :param embed_size: size of image label embedded vector
        :param z_dim: input noise vector
        :param latent_size: size of feature vector
        :param img_channel: an output image channel (default=3)
        :param gen_dim: number of dimension factor (default=64)
        """
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.img_channel = img_channel
        self.net = nn.Sequential(
            # Input: batch x (z_dim + embed_size + feature_size) x 1 x 1 (see definition of noise in below code)
            self._block(z_dim + embed_size + latent_size, gen_dim * 16, 4, 1, 0),  # batch x 1024 x 4 x 4
            self._block(gen_dim * 16, gen_dim * 8, 4, 2, 1),  # batch x 512 x 8 x 8
            self._block(gen_dim * 8, gen_dim * 4, 4, 2, 1),  # batch x 256 x 16 x 16
            self._block(gen_dim * 4, gen_dim * 2, 4, 2, 1),  # batch x 128 x 32 x 32
            nn.ConvTranspose2d(
                gen_dim * 2, self.img_channel, kernel_size=4, stride=2, padding=1,
                # did not use block because the last layer won't use batch norm or relu
            ),  # batch x 3 x 64 x 64
            nn.Tanh(),
            # squeeze output to [-1, 1]; easier to converge.  also will match to our normalize(0.5....) images
        )
        self.embed = nn.Embedding(num_classes, embed_size)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,  # batch norm does not require bias
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)  # in_place = True
        )

    def forward(self, zz, labels, semantic_latent):
        """
        :param zz: noise vector in shape (bs, self.z_dim, 1, 1)
        :param labels: label tensor where store the number of class which in shape of (bs, 1)
        :param semantic_latent: (bs, self.feature_size)
        :return: gen_img in shape (bs, 3, 64, 64)
        """
        semantic_latent = semantic_latent.unsqueeze(2).unsqueeze(3)  # batch, feature_size, 1, 1
        embedding = self.embed(labels).unsqueeze(2).unsqueeze(3)  # batch, embed_size, 1, 1
        x = torch.cat([zz, embedding, semantic_latent], dim=1)
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, ngpu, num_classes, image_size=64, latent_size=200, img_channel=3, dis_dim=64):
        """
        :param ngpu:
        :param num_classes: number of image class (use for do label embedding)
        :param image_size: size of input image also used for specify embedding size (64 is recommended)
        :param latent_size: size of latent vector
        :param img_channel: number of image channel
        :param dis_dim: number of dimension factor
        """
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.img_channel = img_channel
        self.image_size = image_size if type(image_size) is tuple else (image_size, image_size)
        self.latent_joining = nn.Sequential(
            nn.Linear(latent_size, image_size * image_size)
        )
        self.net = nn.Sequential(
            # no batch norm in the first layer
            # Input: batch x num_channel x 64 x 64
            # <-----changed num_channel + 1 since we add the labels
            nn.Conv2d(
                self.img_channel + 2, dis_dim, kernel_size=4, stride=2, padding=1,
            ),  # batch x 64 x 32 x 32
            nn.LeakyReLU(0.2, inplace=True),
            self._block(dis_dim, dis_dim * 2, 4, 2, 1),  # batch x 128 x 16 x 16
            self._block(dis_dim * 2, dis_dim * 4, 4, 2, 1),  # batch x 256 x 8 x 8
            self._block(dis_dim * 4, dis_dim * 8, 4, 2, 1),  # batch x 512 x 4  x 4
            nn.Conv2d(dis_dim * 8, 1, kernel_size=4, stride=2, padding=0),  # batch x 1 x 1 x 1 for classification
            #             nn.Sigmoid(), #<------removed!
        )
        self.embed = nn.Embedding(num_classes, self.image_size[0] * self.image_size[1])

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,  # batch norm does not require bias
            ),
            nn.InstanceNorm2d(out_channels, affine=True),  # <----changed here
            nn.LeakyReLU(0.2, True)  # slope = 0.2, in_place = True
        )

    def forward(self, img, labels, semantic_latent):
        """
        :param img: input image in expected shape of (bs, 3, self.image_size[0], self.image_size[1])
        :param labels: label in expected shape of (bs, self.num_classes)
        :param semantic_latent: a latent vector in shape of (bs, self.latent_size)
        :return: The discriminator's judge result
        """
        feature_plate = self.latent_joining(semantic_latent).view(semantic_latent.shape[0], 1, self.image_size[0],
                                                                  self.image_size[1])
        embedding = self.embed(labels).view(labels.shape[0], 1, self.image_size[0], self.image_size[1])

        # feature_em = self.embed_feature(feature).view(feature.shape[0], 1, image_size, image_size)
        x = torch.cat([img, feature_plate, embedding], dim=1)  # batch x (C + 1) x W x H
        return self.net(x)


def gradient_penalty(dis, features, labels, real, fake, device="cpu"):  # <---add labels
    bs, c, h, w = real.shape
    alpha = torch.rand((bs, 1, 1, 1)).repeat(1, c, h, w).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)

    # Calculate dis scores
    mixed_scores = dis(interpolated_images, labels, features)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gp = torch.mean((gradient_norm - 1) ** 2)
    return gp
