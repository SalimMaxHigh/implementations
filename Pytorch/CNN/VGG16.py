from torch import nn
import torch

batch_size = 256
momentum = 0.9
lr = 10**-2

model = nn.Sequential(
    nn.Conv2d(3, 64, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(64, 64, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, stride=2),

    nn.Conv2d(64, 128, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(128, 128, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, stride=2),

    nn.Conv2d(128, 256, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(256, 256, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(256, 256, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, stride=2),

    nn.Conv2d(256, 512, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(512, 512, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(512, 512, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, stride=2),

    nn.Conv2d(512, 512, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(512, 512, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(512, 512, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, stride=2),

    nn.Flatten(),
    nn.Linear(512 * 7 * 7, 4096),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 1000),
    nn.Softmax(dim=1)

)

input_img = torch.randn(1,3,224,224)
result = model(input_img)
print(result.size())