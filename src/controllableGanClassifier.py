import torch.nn as nn
from torchvision import models
from torchsummary import summary


resnetModel = models.resnet152(weights="ResNet152_Weights.IMAGENET1K_V1")
for param in resnetModel.fc.parameters():
    param.requires_grad = False


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
