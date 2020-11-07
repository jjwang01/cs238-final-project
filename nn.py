import torch
import torch.nn as nn
import torch.nn.functional as F 

class QLNN(nn.Module):
    def __init__(self, dims):
        super().__init__()

        self.dims = dims


        self.fc1 = nn.Linear(dims[0], dims[1])
        nn.init.kaiming_normal_(self.fc1.weight)
        self.fc2 = nn.Linear(dims[1], dims[2])
        nn.init.kaiming_normal_(self.fc2.weight)
        self.fc3 = nn.Linear(dims[2], dims[3])
        nn.init.kaiming_normal_(self.fc3.weight)


    def forward(self, x):
        # print("hi1")
        # print(x_torch.shape)
        # conv_out1 = F.relu(self.conv1(x))
        # h1 = torch.relu(self.fc1(x.t()))
        # print(x.shape)
        # flatten = self.flatten(x)
        # print(flatten.shape)

        h1 = F.relu(self.fc1(x.flatten()))
        # print("hi2")
        h2 = F.relu(self.fc2(h1))
        h3 = F.softmax(self.fc3(h2))

        # print("hi3")
        # h3 = self.fc3(h2)
        # print("hi4")
        scores = h3
        # print(scores)
        # print(scores.shape)
        # scores = self.fc3(F.tanh(self.fc2(F.tanh(self.fc1(x)))))

        return scores






