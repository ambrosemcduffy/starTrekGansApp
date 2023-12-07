import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset
from PIL import Image
from scipy.linalg import sqrtm

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import truncnorm


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.label_map = self._create_label_map()

        # Lists to store file names and labels
        self.file_names = []
        self.labels = []

        # Recursively get all image file paths from root_dir and its subdirectories
        for dp, dn, filenames in os.walk(root_dir):
            for f in filenames:
                if f.lower().endswith(("png", "jpg", "jpeg")):
                    self.file_names.append(os.path.join(dp, f))
                    # Extract folder name from the directory path and map to label
                    folder_name = os.path.basename(dp)
                    self.labels.append(self.label_map[folder_name])

    def _create_label_map(self):
        # This method creates a dynamic label_map dictionary
        folders = [
            d
            for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d))
        ]
        return {folder: idx for idx, folder in enumerate(folders)}

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_name = self.file_names[idx]
        label = self.labels[idx]
        image = Image.open(img_name).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def calculate_fid(act1, act2):
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    sum_square_diff = np.sum((mu1 - mu2) ** 2)
    cov_mean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real
    fid = sum_square_diff + np.trace(sigma1 + sigma2 - 2.0 * cov_mean)
    return fid


def gradient_penalty(disc, real, fake, device="cpu"):
    batch_size, C, H, W = real.shape
    epsilon = torch.rand((batch_size, 1, 1, 1)).expand_as(real).to(device)
    interpolated_images = epsilon * real + (1 - epsilon) * fake
    interpolated_images.requires_grad_(True)

    # Get the critic's scores for the interpolated images
    interpolated_scores = disc(interpolated_images)

    # Calculate the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        outputs=interpolated_scores,
        inputs=interpolated_images,
        grad_outputs=torch.ones_like(interpolated_scores).to(device),
        create_graph=True,
        retain_graph=True,
    )[0]

    # Compute the gradient penalty
    gradient = gradient.view(gradient.size(0), -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()
    return gradient_penalty


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def truncated_normal(size, threshold=2.0, dtype=torch.FloatTensor, volatile=False):
    values = truncnorm.rvs(-threshold, threshold, size=size)
    values = dtype(values)
    if volatile:
        values = values.volatile()
    return values


def apply_spectral_norm(module):
    """Applies spectral normalization to a given module."""
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        return nn.utils.spectral_norm(module)
    return module


def getConcatedInput(inputData, labels, nClasses):
    oneHot = F.one_hot(labels, nClasses)
    oneHot_expanded = (
        oneHot.unsqueeze(2)
        .unsqueeze(3)
        .expand(-1, -1, inputData.size(2), inputData.size(3))
    )
    return torch.cat((inputData, oneHot_expanded), dim=1)
