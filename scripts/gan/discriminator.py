import torch.nn as nn


class PatchDiscriminator(nn.Module):

    def __init__(self, in_chans=2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_chans, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, x):
        return self.net(x)