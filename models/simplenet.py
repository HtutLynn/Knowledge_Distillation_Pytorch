import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class simple_conv_net(nn.Module):
    """
    Build a simple convolutional neural network in Pytorch.
    """

    def __init__(self):
        """
        We define an convolutional network that predicts the class of an image. The components
        required are:

        - convolution: applying convolution operation on the input image data or feature maps returned from the pre-convolution process
        - fc: a fully connected layer that maps the feature maps/feature embeddings discrete outputs
        Args:
            params: (Params) contains num_channels # not included in this implementation
        """
        super(simple_conv_net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(5,5))
        self.batchnoorm1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=((5,5)))
        self.batchnoorm2 = nn.BatchNorm2d(16)
        self.fc1   = nn.Linear(16 * 5 * 5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, input_data):
        """
        Perform the inferencing/forward pass of the model using input_data to map to desired outputs
        :params input_data: A 4D Tensor in (batch_size * width * height * channels) format
        """
        x = F.relu(self.batchnoorm1(self.conv1(input_data)))
        x = self.pool(x)
        x = F.relu(self.batchnoorm2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x