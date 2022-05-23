import torch
from torch import nn


# Component stuff
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, inp):
        shortcut = self.shortcut(inp)
        inp = nn.ReLU()(self.bn1(self.conv1(inp)))
        inp = nn.ReLU()(self.bn2(self.conv2(inp)))
        inp = inp + shortcut
        return nn.ReLU()(inp)


class ReshaperBlock(nn.Module):
    def __init__(self, height, width, output_dim):
        super(ReshaperBlock, self).__init__()
        self.ts = [-1] + ([1] * (output_dim - 2)) + [height, width]  # target shape

    def forward(self, x):
        return x.reshape(self.ts)


class ShallowResNet18Extractor(nn.Module):
    """
    This extractor model based on ResNet18
    """

    def __init__(self, in_features, out_features, num_classes, dropout_p=0.0):
        super(ShallowResNet18Extractor, self).__init__()
        dp = dropout_p

        self.shaper = nn.Sequential(
            nn.Linear(in_features, 1600),
            ReshaperBlock(40, 40, 3),
            nn.ReLU(),
            nn.Conv2d(kernel_size=1, in_channels=1, out_channels=3),
            nn.BatchNorm2d(3)
        )

        self.l0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),  # w,h div by 2 -> since stride=2
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # w,h div by 2 -> since stride=2
            nn.BatchNorm2d(64),
            nn.ReLU()
        )  # Hence, this block will return in shape of (ch=64, w/4, h/4)

        self.l1 = nn.Sequential(
            ResBlock(64, 64, False),
            ResBlock(64, 64, False),
            ResBlock(64, 64, False),
            nn.Dropout2d(p=dp)
        )

        self.l2 = nn.Sequential(
            ResBlock(64, 128, True),
            ResBlock(128, 128, False),
            ResBlock(128, 128, False),
            ResBlock(128, 128, False),
            nn.Dropout2d(p=dp)
        )

        self.l3 = nn.Sequential(
            ResBlock(128, 256, True),
            ResBlock(256, 256, False),
            ResBlock(256, 256, False),
            ResBlock(256, 256, False),
            ResBlock(256, 256, False),
            ResBlock(256, 256, False),
            nn.Dropout2d(p=dp)
        )

        self.l4 = nn.Sequential(
            ResBlock(256, 512, True),
            ResBlock(512, 512, False),
            ResBlock(512, 512, False),
            nn.Dropout2d(p=dp)
        )

        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(128, out_features),
            nn.ReLU()
        )
        self.final_fc = nn.Linear(out_features, num_classes)

    def forward(self, x):
        x = self.shaper(x)
        x = self.l0(x)
        x = self.l1(x)
        x = self.l2(x)
        # x = self.l3(x)
        # x = self.l4(x)
        x = self.gap(x)
        x = torch.flatten(x, start_dim=1)
        latent = self.fc(x)
        class_p = self.final_fc(latent)
        return latent, class_p


class ResNet18Extractor(nn.Module):
    """
    This extractor model based on ResNet18
    """

    def __init__(self, in_features, out_features, num_classes, dropout_p=0.0):
        super(ResNet18Extractor, self).__init__()
        dp = dropout_p

        self.shaper = nn.Sequential(
            nn.Linear(in_features, 1600),
            ReshaperBlock(40, 40, 3),
            nn.ReLU(),
            nn.Conv2d(kernel_size=1, in_channels=1, out_channels=3),
            nn.BatchNorm2d(3)
        )

        self.l0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),  # w,h div by 2 -> since stride=2
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # w,h div by 2 -> since stride=2
            nn.BatchNorm2d(64),
            nn.ReLU()
        )  # Hence, this block will return in shape of (ch=64, w/4, h/4)

        self.l1 = nn.Sequential(
            ResBlock(64, 64, False),
            ResBlock(64, 64, False),
            ResBlock(64, 64, False),
            nn.Dropout2d(p=dp)
        )

        self.l2 = nn.Sequential(
            ResBlock(64, 128, True),
            ResBlock(128, 128, False),
            ResBlock(128, 128, False),
            ResBlock(128, 128, False),
            nn.Dropout2d(p=dp)
        )

        self.l3 = nn.Sequential(
            ResBlock(128, 256, True),
            ResBlock(256, 256, False),
            ResBlock(256, 256, False),
            ResBlock(256, 256, False),
            ResBlock(256, 256, False),
            ResBlock(256, 256, False),
            nn.Dropout2d(p=dp)
        )

        self.l4 = nn.Sequential(
            ResBlock(256, 512, True),
            ResBlock(512, 512, False),
            ResBlock(512, 512, False),
            nn.Dropout2d(p=dp)
        )

        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(128, out_features),
            nn.ReLU()
        )
        self.final_fc = nn.Linear(out_features, num_classes)

    def forward(self, x):
        x = self.shaper(x)
        x = self.l0(x)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.gap(x)
        x = torch.flatten(x, start_dim=1)
        latent = self.fc(x)
        class_p = self.final_fc(latent)
        return latent, class_p


class ResNet34Extractor(nn.Module):
    """
    This extractor model based on ResNet34
    """

    def __init__(self, in_features, out_features, num_classes, dropout_p=0.0):
        super(ResNet34Extractor, self).__init__()
        dp = dropout_p

        self.shaper = nn.Sequential(
            nn.Linear(in_features, 1600),
            ReshaperBlock(40, 40, 3),
            nn.ReLU(),
            nn.Conv2d(kernel_size=1, in_channels=1, out_channels=3),
            nn.BatchNorm2d(3)
        )

        self.l0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),  # w,h div by 2 -> since stride=2
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # w,h div by 2 -> since stride=2
            nn.BatchNorm2d(64),
            nn.ReLU()
        )  # Hence, this block will return in shape of (ch=64, w/4, h/4)

        self.l1 = nn.Sequential(
            ResBlock(64, 64, False),
            ResBlock(64, 64, False),
            ResBlock(64, 64, False),
            nn.Dropout2d(p=dp)
        )

        self.l2 = nn.Sequential(
            ResBlock(64, 128, True),
            ResBlock(128, 128, False),
            ResBlock(128, 128, False),
            ResBlock(128, 128, False),
            nn.Dropout2d(p=dp)
        )

        self.l3 = nn.Sequential(
            ResBlock(128, 256, True),
            ResBlock(256, 256, False),
            ResBlock(256, 256, False),
            ResBlock(256, 256, False),
            ResBlock(256, 256, False),
            ResBlock(256, 256, False),
            nn.Dropout2d(p=dp)
        )

        self.l4 = nn.Sequential(
            ResBlock(256, 512, True),
            ResBlock(512, 512, False),
            ResBlock(512, 512, False),
            nn.Dropout2d(p=dp)
        )

        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(128, out_features),
            nn.ReLU()
        )
        self.final_fc = nn.Linear(out_features, num_classes)

    def forward(self, x):
        x = self.shaper(x)
        x = self.l0(x)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.gap(x)
        x = torch.flatten(x, start_dim=1)
        latent = self.fc(x)
        class_p = self.final_fc(latent)
        return latent, class_p
