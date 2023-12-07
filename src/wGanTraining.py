import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.utils as vutils

from torchvision import transforms
from torch.optim import RMSprop
from torchvision.models import inception_v3
from torch.utils.tensorboard import SummaryWriter


from wGanModel import Generator, Discriminator
from ganUtils import (
    weights_init,
    CustomDataset,
    gradient_penalty,
    truncated_normal,
    calculate_fid,
)

writer = SummaryWriter("runs/wganStarTrekCharactersTraining")
torch.cuda.empty_cache()

# Preprocessing Data
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((64, 64), antialias=True),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


# HyperParams some of this I've taken from the paper itself.
zDim = 512
gHiddenDim = 128
dHidden = 64
lr = 0.0002
batchSize = 64
lambda_gp = 5
lambda_iteration = 5
n_epochs = 2000

# I'm loading in the inception model to perform Frechet Inception Distance FID
inception_model = inception_v3(
    weights="Inception_V3_Weights.IMAGENET1K_V1", transform_input=False
).cuda()
inception_model = inception_model.eval()
gen = Generator(zDim=zDim, hiddenDim=gHiddenDim)
disc = Discriminator(imDim=3, hiddenDim=dHidden)


if torch.cuda.is_available():
    print("Found Cuda Device!:\n{}".format(torch.cuda.get_device_name()))
    gen = gen.cuda()
    disc = disc.cuda()


optimizerDisc = RMSprop(disc.parameters(), lr=lr)
optimizerGen = RMSprop(gen.parameters(), lr=lr)
criterion = nn.BCEWithLogitsLoss()

dataDirectory = "/mnt/e/gansStudy/starTrekFaces/"
custom_data = CustomDataset(root_dir=dataDirectory, transform=transform)
trainloader = torch.utils.data.DataLoader(
    custom_data, batch_size=batchSize, shuffle=True, num_workers=24
)

# initializing out weights..
gen.apply(weights_init)
disc.apply(weights_init)

gen_losses = []
disc_losses = []
fidScores = []

# This maybe weird and I should have higherez Images, but I don't so I'm resizing for FID
resize_transform = transforms.Resize(
    (299, 299), antialias=True
)  # Define the resize transform

# Train loop
for e in range(n_epochs):
    genLossTotal = 0.0
    discLossTotal = 0.0
    num_batches = 0
    for images, labels in trainloader:
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()

        # For wgan we sometimes update the critic multiple times.
        for _ in range(lambda_iteration):
            disc.zero_grad()
            noise = torch.randn(images.size(0), zDim, 1, 1)
            if torch.cuda.is_available():
                noise = noise.cuda()

            # This might look weird, and it is but I'm using the truncation trick to filter lower fedelity images..
            noise = truncated_normal(
                size=(noise.size(0), zDim, 1, 1),
                threshold=6.5,
                dtype=torch.cuda.FloatTensor
                if torch.cuda.is_available()
                else torch.FloatTensor,
            )
            fakeImage = gen.forward(noise)

            discPredFake = disc.forward(fakeImage)
            discPredReal = disc.forward(images)

            # Compute WGAN loss with gradient penalty
            wgan_loss = discPredReal.mean() - discPredFake.mean()
            gp = gradient_penalty(disc, images, fakeImage, device=images.device)
            lossDisc = -wgan_loss + lambda_gp * gp
            discLossTotal += lossDisc.item()
            lossDisc.backward()
            optimizerDisc.step()

        # Update the generator
        gen.zero_grad()
        genfakeImage = gen.forward(noise.cuda())
        genPred = disc.forward(genfakeImage)
        genLoss = -genPred.mean()
        genLossTotal += genLoss.item()
        genLoss.backward()
        optimizerGen.step()

    if e % 1 == 0:  # Calculate every 10 epochs
        real_features = []
        fake_features = []
        with torch.no_grad():
            for images, labels in trainloader:
                images = images.cuda()
                labels = labels.cuda()
                # Apply the resize transform to real images
                resized_real_images = torch.stack(
                    [resize_transform(image) for image in images]
                )
                real_features.append(inception_model(resized_real_images).cpu().numpy())

                fake_images = gen(noise)

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
        torch.save(gen.state_dict(), "generator_all_wganGPV2.pth")
        gen_losses.append(genLossTotal)
        disc_losses.append(discLossTotal)
        writer.add_scalars(
            "GAN_Losses", {"Generator": genLossTotal, "Discriminator": discLossTotal}, e
        )
        writer.add_scalar("Losses/FID", fid_score, e)
        print(e, genLossTotal, discLossTotal)
        fake_image = genfakeImage[0].detach().cpu()  # Take the first image in the batch
        fake_image = (fake_image + 1) / 2  # Unnormalize the image
        fake_image = fake_image.permute(
            1, 2, 0
        )  # If the image is grayscale, this will remove the singleton color channel dimension
        img_grid_input = vutils.make_grid(genfakeImage[:4], normalize=True)
        writer.add_image("input_images", img_grid_input, global_step=e)
