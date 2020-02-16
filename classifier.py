from torch import nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        '''
        1 input layer
        5 output layer
        Kernel size 3x3
        '''
        self.conv1 = nn.Conv2d(1, 5, 3)

        # Kernel size 2x2
        self.pool = nn.MaxPool2d(2)

        '''
        5 input layers
        15 output layers
        Kernel size 3x3
        '''
        self.conv2 = nn.Conv2d(5, 15, 3)

        # Flatten into fully connected layer
        self.fc1 = nn.Linear(15 * 5 * 5, 120)

        self.fc2 = nn.Linear(120, 60)

        # 10 output classes
        self.fc3 = nn.Linear(60, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # Reshape, -1 is inferred from other dimensions
        x = x.view(-1, 15 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x


