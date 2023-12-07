import torch
from torch.optim import Adam
from wGanModel import Generator
from controllableGanClassifier import Classifier
import matplotlib.pyplot as plt
from torchvision import models
from ganUtils import truncated_normal
from scipy.stats import truncnorm


# Load your pre-trained classifier
classifier = models.resnet152(pretrained=False)
numClasses = 7
classifier.fc = torch.nn.Linear(classifier.fc.in_features, numClasses)
classifier.load_state_dict(torch.load("classifier.pth"))
gen = Generator(512)


gen.load_state_dict(
    torch.load("/mnt/e/gansStudy/generator_all_wganGPV2_739_greatData.pth")
)
classifier.load_state_dict(torch.load("classifier.pth"))
gen.eval()
classifier.eval()


if torch.cuda.is_available():
    classifier = classifier.cuda()
    gen = gen.cuda()


def get_noise(batchSize, zDim, device="cpu"):
    return torch.randn(batchSize, zDim, 1, 1).to(device)


def calculate_updated_noise(noise, weight):
    return noise + (noise.grad * weight)


# def get_score(current_classifications, original_classifications, target_indicies, other_indices, penalty_weight):
#     diff_ = current_classifications[:, other_indices] - original_classifications[:other_indices]
#     norms_ = torch.norm(diff_, dim=1)
#     other_class_penalty = -(norms_.mean() * penalty_weight)
#     target_score = current_classifications[:, target_indicies].mean()
#     return target_score + other_class_penalty


def get_score(
    current_classifications,
    original_classifications,
    target_indicies,
    other_indices,
    penalty_weight,
):
    # Convert the list of booleans to a boolean tensor
    other_indices_tensor = torch.tensor(
        other_indices, device=current_classifications.device
    )

    # Index with boolean tensor
    diff_ = (
        current_classifications[:, other_indices_tensor]
        - original_classifications[:, other_indices_tensor]
    )
    norms_ = torch.norm(diff_, dim=1)
    other_class_penalty = -(norms_.mean() * penalty_weight)
    target_score = current_classifications[:, target_indicies].mean()
    return target_score + other_class_penalty


learnRate = 0.001
opt = Adam(classifier.fc.parameters(), lr=learnRate, betas=(0.9, 0.999), eps=1e-8)
noise = torch.randn(32, 100, 1, 1, device="cuda")
noise = noise.cuda()


# First generate a bunch of images with the generator
# torch.manual_seed(30)

n_images = 8
fake_image_history = []
grad_steps = 100  # Number of gradient steps to take
skip = 5  # Number of gradient steps to skip in the visualization
feature_names = ["Borg", "Data", "LaForge", "Kirk", "Picard", "Riker", "Worf"]
target_indices = feature_names.index("Riker")


noise = get_noise(32, 512, "cuda")

# noise.requires_grad_()
# for i in range(grad_steps):
#     print(i)
#     opt.zero_grad()
#     fake = gen(noise)
#     fake_image_history.append(fake.detach())
#     fake_classes_score = classifier(fake)[:, target_indices].mean()
#     print(i, fake_classes_score.item())
#     fake_classes_score.backward()
#     noise.data = calculate_updated_noise(noise, 1 / grad_steps)
#     plt.rcParams['figure.figsize'] = [n_images * 2, grad_steps * 2]


feature_names = ["Borg", "Data", "LaForge", "Kirk", "Picard", "Riker", "Worf"]
target_indices = feature_names.index("Kirk")
other_indices = [cur_idx != target_indices for cur_idx, _ in enumerate(feature_names)]
original_classifications = classifier(gen(noise)).detach()

print(" Selected", target_indices, feature_names[target_indices])

n_images = 8
fake_image_history = []
grad_steps = 1260  # Number of gradient steps to take
skip = 100  # Number of gradient steps to skip in the visualization
noise = torch.randn(128, 512, 1, 1, device="cuda")
original_classifications = classifier(gen(noise)).detach()
noise = truncated_normal(
    size=(128, 512, 1, 1),
    threshold=4,
    dtype=torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor,
)
noise.requires_grad_()
for i in range(grad_steps):
    print(i)
    opt.zero_grad()
    fake = gen(noise)
    fake_image_history += [fake]
    fake_classes_score = get_score(
        classifier(fake),
        original_classifications,
        target_indices,
        other_indices,
        penalty_weight=0.01,
    )
    print(i, fake_classes_score.item())
    fake_classes_score.backward()
    noise.data = calculate_updated_noise(noise, 1 / grad_steps)
    plt.rcParams["figure.figsize"] = [n_images * 2, grad_steps * 2]

skip_steps = int(grad_steps * 0.25)

# Adjust the number of subplots accordingly
fig, axs = plt.subplots(
    grad_steps // skip_steps,
    n_images,
    figsize=(n_images * 2, (grad_steps // skip_steps) * 2),
)

for i in range(0, grad_steps, skip_steps):
    for j in range(n_images):
        # Adjust for 1-dimensional axs array
        if grad_steps // skip_steps == 1:
            ax = axs[j]
        else:
            ax = axs[i // skip_steps, j]

        ax.imshow(
            (fake_image_history[i][j].permute(1, 2, 0).cpu().detach().numpy() * 0.5)
            + 0.5
        )
        ax.axis("off")

plt.tight_layout()
plt.show()
