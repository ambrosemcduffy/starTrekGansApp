import torch
import torch.nn as nn
import torch.nn.functional as F
from ganUtils import apply_spectral_norm


class Discriminator(nn.Module):
    def __init__(self, imDim=3, hiddenDim=16):
        super(Discriminator, self).__init__()
        self.conv1 = apply_spectral_norm(nn.Conv2d(imDim, hiddenDim, (4, 4), 2, 1))
        self.conv2 = apply_spectral_norm(
            nn.Conv2d(hiddenDim, hiddenDim * 2, (4, 4), 2, 1)
        )
        self.conv3 = apply_spectral_norm(
            nn.Conv2d(hiddenDim * 2, hiddenDim * 4, (4, 4), 2, 1)
        )
        self.fc1 = apply_spectral_norm(nn.Linear(8 * 8 * (hiddenDim * 4), 1))

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


# Added an Extra layer to the Generator
class Generator(nn.Module):
    def __init__(self, zDim=100, hiddenDim=128):
        super(Generator, self).__init__()
        self.tConv1 = nn.ConvTranspose2d(
            zDim, hiddenDim * 8, (4, 4), 1, 0
        )  # Output size: 4x4
        self.bn1 = nn.BatchNorm2d(hiddenDim * 8)
        self.tConv2 = nn.ConvTranspose2d(
            hiddenDim * 8, hiddenDim * 4, (4, 4), 2, 1
        )  # Output size: 8x8
        self.bn2 = nn.BatchNorm2d(hiddenDim * 4)
        self.tConv3 = nn.ConvTranspose2d(
            hiddenDim * 4, hiddenDim * 2, (4, 4), 2, 1
        )  # Output size: 16x16
        self.bn3 = nn.BatchNorm2d(hiddenDim * 2)
        self.tConv4 = nn.ConvTranspose2d(
            hiddenDim * 2, hiddenDim, (4, 4), 2, 1
        )  # Output size: 32x32
        self.bn4 = nn.BatchNorm2d(hiddenDim)
        self.tConv5 = nn.ConvTranspose2d(
            hiddenDim, 3, (4, 4), 2, 1
        )  # Output size: 64x64

    def forward(self, x):
        x = self.tConv1(x)
        x = F.relu(self.bn1(x))
        x = self.tConv2(x)
        x = F.relu(self.bn2(x))
        x = self.tConv3(x)
        x = F.relu(self.bn3(x))
        x = self.tConv4(x)
        x = F.relu(self.bn4(x))
        x = torch.tanh(self.tConv5(x))
        return x
