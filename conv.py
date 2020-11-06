import torch
import torch.nn as nn
import torch.nn.functional as F 

BOARD_SIZE = 8

class QLConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        # let's say that the total number of actions is the size of the board and deal with restrictions later
        self.num_actions = BOARD_SIZE**2

        # input is (1, num_actions, 8, 8,)
        # h = (8 - (k-1) - 1) / 2 + 1 = (8-k)/2 + 1
        self.conv1 = nn.Conv2d(in_channels=self.num_actions, out_channels=128, kernel_size=4, stride=2)
        nn.init.kaiming_normal_(self.conv1.weight)
        # (128, 3, 3)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, stride=1)
        nn.init.kaiming_normal_(self.conv2.weight)
        # h = (3 - (k-1) - 1) / 1 + 1 = 4-k
        # (128, 2, 2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=2, stride=1)
        nn.init.kaiming_normal_(self.conv3.weight)
        # h = (2 - (k-1) - 1) / 1 + 1 = 3-k
        # (256, 1, 1)
        
        self.flatten = nn.Flatten()
        # (256,)
        self.fc1 = nn.Linear(in_features=256, out_features=128)
        nn.init.kaiming_normal_(self.fc1.weight) 
        # (128,)
        self.fc2 = nn.Linear(in_features=128, out_features=self.num_actions)
        nn.init.kaiming_normal_(self.fc2.weight)


    def forward(self, x):

        conv_out1 = F.relu(self.conv1(x))
        conv_out2 = F.relu(self.conv2(conv_out1))
        conv_out3 = F.relu(self.conv3(conv_out2))

        flatten_out = self.flatten(conv_out3)
        fc_out1 = F.relu(self.fc1(flatten_out))
        fc_out2 = F.softmax(self.fc2(fc_out1))

        return fc_out2






