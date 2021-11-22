from torchvision import models
import torch.nn as nn


model = models.vgg16(pretrained=True)


if __name__ == "__main__":
# Freeze model weights
    for param in model.parameters():
        param.requires_grad = False


# Add last layer per Waterloo paper:
    model.classifier[6] = nn.Sequential(
        nn.Linear(4096, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256,2),
        nn.LogSoftmax(dim=1)
        
        )

    print(model.classifier)


model = model.to('cuda')

