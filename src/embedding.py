import torch
from torchvision import models, transforms
from PIL import Image

# Load the pre-trained Inception V3 model and replace the fully connected layer
inception = models.inception_v3(weights="Inception_V3_Weights.DEFAULT")
inception.fc = torch.nn.Identity()
inception.eval()

# Example for loading and preprocessing an image
image_path = "/home/ambrosemcduffy/autoEncoder/images/picardDataset5K/picard0000.jpg"
image = Image.open(image_path)

# Preprocess the image
preprocess = transforms.Compose(
    [
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0)  # Add a batch dimension

# Check if a GPU is available and if not, use a CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move the input and model to the correct device
input_batch = input_batch.to(device)  # Use the `to` method here
inception = inception.to(device)

# Calculate the embeddings
embedding = None
with torch.no_grad():
    embedding = inception(input_batch)

print(embedding.shape)
