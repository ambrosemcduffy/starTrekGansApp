import sys
import numpy as np
import torch

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import truncnorm


import torch.nn as nn
import torch
import torch.nn.functional as F
import os

from torch.optim import Adam
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms, models
from torch.utils.data import random_split
from wGanModel import Generator

from torchsummary import summary


from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QSlider
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QPushButton, QLineEdit


class Classifier(nn.Module):
    def __init__(self, numClasses):
        super(Classifier, self).__init__()
        self.resnet = self.getResnet()
        self.fc = nn.Linear(2048, numClasses)

    def getResnet(self):
        resnetModel = models.resnet152(weights="ResNet152_Weights.IMAGENET1K_V1")
        modules = list(resnetModel.children())[:-1]
        resnetModel = nn.Sequential(*modules)

        for param in resnetModel.parameters():
            param.requires_grad = False
        return resnetModel

    def forward(self, x):
        x = self.resnet.forward(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


gen = Generator(zDim=512)
model = Classifier(numClasses=7)

if torch.cuda.is_available():
    gen = gen.cuda()
    model = model.cuda()

# gen.load_state_dict(torch.load('/mnt/e/gansStudy/generator_all_wganGPV2_739_greatData.pth'))
gen.load_state_dict(
    torch.load("/mnt/e/gansStudy/weights/generator_all_wganGPV2Step800_GoodRiker.pth")
)

zDim = 512

gen.eval()


def truncated_normal(size, threshold=2.0, dtype=torch.FloatTensor, volatile=False):
    values = truncnorm.rvs(-threshold, threshold, size=size)
    values = dtype(values)
    if volatile:
        values = values.volatile()
    return values


class GANVisualizer(QWidget):
    def __init__(self):
        super().__init__()

        self.all_images = self.generate_images()

        layout = QVBoxLayout()

        # Add widgets for seed and truncation thresholds
        self.seed_input = QLineEdit(self)
        self.seed_input.setPlaceholderText("Enter seed (default=random)")
        layout.addWidget(self.seed_input)

        self.truncation_input = QLineEdit(self)
        self.truncation_input.setPlaceholderText(
            "Enter truncation threshold (default=3.0)"
        )
        layout.addWidget(self.truncation_input)

        self.generate_button = QPushButton("Generate New Images", self)
        self.generate_button.clicked.connect(self.generate_new_images)
        layout.addWidget(self.generate_button)

        self.label = QLabel(self)
        layout.addWidget(self.label)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(len(self.all_images) - 1)
        self.slider.valueChanged.connect(self.update_image)
        layout.addWidget(self.slider)

        self.setLayout(layout)

        self.update_image(0)  # Display the first image by default

    def generate_new_images(self):
        seed = self.seed_input.text()
        truncation = self.truncation_input.text()

        # Use the provided seed and truncation or use default values
        seed = int(seed) if seed else np.random.randint(1e6)
        truncation = float(truncation) if truncation else 3.0

        np.random.seed(seed)
        torch.manual_seed(seed)

        self.all_images = self.generate_images(truncation_threshold=truncation)
        self.slider.setMaximum(len(self.all_images) - 1)
        self.update_image(0)

    def generate_images(self, truncation_threshold=3.0):
        # Define starting and ending noise vectors
        z1 = truncated_normal(
            size=(1, zDim, 1, 1),
            threshold=truncation_threshold,
            dtype=torch.cuda.FloatTensor
            if torch.cuda.is_available()
            else torch.FloatTensor,
        )
        z2 = truncated_normal(
            size=(1, zDim, 1, 1),
            threshold=truncation_threshold,
            dtype=torch.cuda.FloatTensor
            if torch.cuda.is_available()
            else torch.FloatTensor,
        )

        # Pre-generate all images
        num_steps = 64
        all_images = []

        for alpha in np.linspace(0, 1, num_steps):
            noise_interp = alpha * z1 + (1 - alpha) * z2

            with torch.no_grad():
                gen_image = gen(noise_interp.cuda()).cpu().view(3, 64, 64)

            gen_image = (gen_image + 1) / 2
            all_images.append(gen_image.permute(1, 2, 0).numpy())

        return all_images

    def update_image(self, slider_value):
        img = self.all_images[slider_value]

        # Convert image data to [0, 255] and uint8 type
        img = (img * 255).astype(np.uint8)

        height, width, channel = img.shape
        bytes_per_line = 3 * width
        q_img = QImage(
            img.tobytes(), width, height, bytes_per_line, QImage.Format_RGB888
        )

        pixmap = QPixmap.fromImage(q_img)

        # Scale the pixmap
        scaled_pixmap = pixmap.scaled(1024, 1024, Qt.KeepAspectRatio)
        self.label.setPixmap(scaled_pixmap)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GANVisualizer()
    window.show()
    sys.exit(app.exec_())
