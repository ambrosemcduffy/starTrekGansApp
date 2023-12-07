import torch
import numpy as np
import matplotlib.pyplot as plt


from torchvision.transforms import Grayscale
from model import CustomDataset
from torchvision import transforms

# data_dir = '/mnt/e/autoEncoder/images/denoise_and_color/'
# transforms = transforms.Compose([transforms.Resize((140, 244)),
#                                  transforms.ToTensor()])


data_dir = "images/picardDataset5K/"
input_transforms = transforms.Compose(
    [Grayscale(num_output_channels=3), transforms.ToTensor()]
)

target_transforms = transforms.Compose([transforms.ToTensor()])

batchsize = 32
dataset = CustomDataset(
    root_dir=data_dir,
    input_transform=input_transforms,
    target_transform=target_transforms,
)
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batchsize, shuffle=True, num_workers=24
)


images, labels = next(iter(dataloader))
images = images.numpy()
labels = labels.numpy()

labels = np.transpose(labels, (0, 2, 3, 1))
images = np.transpose(images, (0, 2, 3, 1))


fig = plt.figure(figsize=(10, 5))
columns = 4
rows = 2

fig, axs = plt.subplots(rows, columns * 2, figsize=(10, 5))

for i in range(rows):
    for j in range(columns):
        index = i * columns + j

        # Plotting images on the left side of the subplot
        axs[i, j * 2].imshow(images[index], cmap="gray")
        axs[i, j * 2].axis("off")  # to hide the axes

        # Plotting labels on the right side of the subplot
        axs[i, j * 2 + 1].imshow(labels[index])
        axs[i, j * 2 + 1].axis("off")  # to hide the axes

plt.show()
