import torch
from torch.optim import Adam
from torch.utils.data import random_split
from torchvision import transforms, models

from ganUtils import CustomDataset

# Load the pre-trained ResNet152 model
resnetModel = models.resnet152(pretrained=True)

# Freeze all layers
for param in resnetModel.parameters():
    param.requires_grad = False

# Modify the final layer to match the number of classes you have (7 in this case)
numClasses = 7
resnetModel.fc = torch.nn.Linear(resnetModel.fc.in_features, numClasses)

# Check if CUDA is available and move the model to GPU if possible
if torch.cuda.is_available():
    resnetModel = resnetModel.cuda()

# Define transformations
batchsize = 32
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

data_dir = "/mnt/e/gansStudy/starTrekFaces/"
dataset = CustomDataset(root_dir=data_dir, transform=transform)

train_size = int(0.8 * len(dataset))
valid_size = len(dataset) - train_size

train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
trainloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batchsize, shuffle=True, num_workers=24
)
validloader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=batchsize, shuffle=False, num_workers=24
)

# Define the criterion, optimizer, and scheduler
criterion = torch.nn.CrossEntropyLoss()
optimizer = Adam(resnetModel.fc.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

n_epochs = 20

# Training the model
for e in range(n_epochs):
    trainloss = 0.0
    resnetModel.train()
    for images, labels in trainloader:
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()
        output = resnetModel(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        trainloss += loss.item()

    # Update the learning rate
    scheduler.step()

    # Calculate average training loss
    avg_trainloss = trainloss / len(trainloader)
    if e % 5 == 0:
        print(f"Epoch:{e} -- Average trainLoss:{avg_trainloss}")

    # Validation
    with torch.no_grad():
        resnetModel.eval()
        valid_loss = 0.0
        for images, labels in validloader:
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

            output = resnetModel(images)
            loss = criterion(output, labels)
            valid_loss += loss.item()

        avg_validloss = valid_loss / len(validloader)
        print(f"Epoch:{e} -- Average validLoss:{avg_validloss}")

torch.save(resnetModel.state_dict(), "classifier.pth")
