import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import transforms
from IPython.display import clear_output


from torchvision.models import inception_v3


import matplotlib.pyplot as plt
import numpy as np


from torchvision.models import inception_v3
from ganUtils import (
    calculate_fid,
    getConcatedInput,
    truncated_normal,
    CustomDataset,
    weights_init,
)
from snGanModel import Generator, Discriminator

import matplotlib.pyplot as plt
import numpy as np

import torchvision.utils as vutils

from torch.utils.tensorboard import SummaryWriter
from scipy.stats import truncnorm

torch.cuda.empty_cache()
# writer = SummaryWriter("runs/Gans")

# torch.cuda.empty_cache()


# HyperParams
gHiddenDim = 128
dHidden = 64
lr = 0.0002
batchSize = 128
n_epochs = 3000
colorChannels = 3
zDim = 100
classNum = 3

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((64, 64), antialias=True),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

inception_model = inception_v3(pretrained=True, transform_input=False).cuda()
inception_model = inception_model.eval()
gen = Generator(zDim=zDim, hiddenDim=gHiddenDim)
disc = Discriminator(imDim=colorChannels, hiddenDim=dHidden)


if torch.cuda.is_available():
    print("Found Cuda Device!:\n{}".format(torch.cuda.get_device_name()))
    gen = gen.cuda()
    disc = disc.cuda()


optimizerDisc = Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))
optimizerGen = Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))

criterion = nn.BCEWithLogitsLoss()

custom_data = CustomDataset(
    root_dir="/mnt/e/gansStudy/picardAngles/", transform=transform
)
trainloader = torch.utils.data.DataLoader(
    custom_data, batch_size=batchSize, shuffle=True, num_workers=24
)

gen.apply(weights_init)
disc.apply(weights_init)

gen_losses = []
disc_losses = []
fidScores = []
resize_transform = transforms.Resize(
    (299, 299), antialias=True
)  # Define the resize transform

for e in range(n_epochs):
    genLossTotal = 0.0
    discLossTotal = 0.0
    num_batches = 0
    for images, labels in trainloader:
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()

        disc.zero_grad()
        noise = truncated_normal(
            size=(images.size(0), 100, 1, 1),
            threshold=3.5,
            dtype=torch.cuda.FloatTensor
            if torch.cuda.is_available()
            else torch.FloatTensor,
        )

        noiseLabels = getConcatedInput(noise, labels, classNum)
        imageLabels = getConcatedInput(images, labels, classNum)
        if torch.cuda.is_available():
            noise = noise.cuda()
            noiseLabels = noiseLabels.cuda()

        fakeImage = gen.forward(noiseLabels)
        fakeImageLabels = getConcatedInput(fakeImage, labels, classNum)

        discPredFake = disc.forward(fakeImageLabels)
        discPredReal = disc.forward(imageLabels)
        real_loss = criterion(
            discPredReal.view(-1), torch.ones_like(discPredReal.view(-1))
        )
        fake_loss = criterion(
            discPredFake.view(-1), torch.zeros_like(discPredFake.view(-1))
        )
        lossDisc = real_loss + fake_loss

        discLossTotal += lossDisc.item()
        lossDisc.backward(retain_graph=True)
        optimizerDisc.step()

        # Update the generator
        gen.zero_grad()
        genfakeImage = gen.forward(noiseLabels.cuda())
        genPred = disc.forward(fakeImageLabels)
        genLoss = criterion(genPred.view(-1), torch.ones_like(genPred.view(-1)))

        genLossTotal += genLoss.item()
        genLoss.backward()
        optimizerGen.step()

    if e % 1 == 0:  # Calculate every 10 epochs
        real_features = []
        fake_features = []
        with torch.no_grad():
            for images, labels in trainloader:
                images = images.cuda()

                # Apply the resize transform to real images
                resized_real_images = torch.stack(
                    [resize_transform(image) for image in images]
                )
                real_features.append(inception_model(resized_real_images).cpu().numpy())

                fake_images = gen(noiseLabels)

                # Apply the resize transform to fake images
                resized_fake_images = torch.stack(
                    [resize_transform(image) for image in fake_images]
                )
                fake_features.append(inception_model(resized_fake_images).cpu().numpy())

        real_features = np.concatenate(real_features, axis=0)
        fake_features = np.concatenate(fake_features, axis=0)

        fid_score = calculate_fid(real_features, fake_features)
        fidScores.append(fid_score)
        print(f"FID Score for epoch {e}: {fid_score}")

    if e % 1 == 0:
        torch.save(gen.state_dict(), "generator_conditional_geordiWorf_snGanV1.pth")
        gen_losses.append(genLossTotal)
        disc_losses.append(discLossTotal)

        print(e, genLossTotal, discLossTotal)
        fake_image = genfakeImage[0].detach().cpu()  # Take the first image in the batch
        fake_image = (fake_image + 1) / 2  # Unnormalize the image
        fake_image = fake_image.permute(
            1, 2, 0
        )  # If the image is grayscale, this will remove the singleton color channel dimension
        plt.imshow(fake_image)
        plt.axis("off")

        plt.figure(figsize=(10, 5))  # Adjust the figure size as needed
        plt.plot(range(e + 1), gen_losses, label="Generator Loss")
        plt.plot(range(e + 1), disc_losses, label="Critic Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
        plt.plot(range(e + 1), fidScores, label="FID scores")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
        clear_output(wait=True)
