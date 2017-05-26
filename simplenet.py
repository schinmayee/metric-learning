import torch
import torch.nn as nn
import torch.nn.functional as F

class Simplenet(nn.Module):
    def __init__(self, im_len):
        super(Simplenet, self).__init__()
        # size of first fully connected layer
        self.im_len = im_len
        self.h1_len = (self.im_len-4)/2
        self.h2_len = (self.h1_len-4)/2
        self.fc1_len = self.h2_len*self.h2_len*20
        # all the layers
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(self.fc1_len, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, self.fc1_len)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return self.fc2(x)
