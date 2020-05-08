import numpy as numpy
import torch
import torch.optim as optim
import torch.nn as nn
import torch.functional as F

import torchvision
import torchvision.transforms as transforms

from models.simplenet import simple_conv_net


transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

# The dataset that we are going to train the network is : CIFAR-10 dataset

trainset = torchvision.datasets.CIFAR10(root='/home/htut/Desktop/Knowledge_Distillation_Pytorch/datasets', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                           shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR10(root="/home/htut/Desktop/Knowledge_Distillation_Pytorch/datasets", train=False,
                                        download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=4)

classes = ('plane', 'car', 'bird', 'cat', 'deeer',
            'dog', 'frog', 'horse', 'ship', 'truck')

# Create an instance of the model class
net = simple_conv_net()
print("Model Constructed!")

# setup the device to train with
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# place the model on GPU
net.to(device)

# Construct a loss function for the model
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
print("Loss function and optimizer constructed!")

# train procedure

for epoch in range(2):

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs_batch, labels_batch = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize

        outputs = net(inputs_batch)
        loss = criterion(outputs, labels_batch) # cross entropy loss
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999: # print every 2000 mini-batches
            print("[%d, %5d] loss: %.3f" % 
                    (epoch +1, i + 1, running_loss/2000))
            running_loss = 0.0
        
print('Finished Training')

print("Saving the model.....")
save_path = "/home/htut/Desktop/Knowledge_Distillation_Pytorch/checkpoints/net/simple_net_0.1.pt"
torch.save(net.state_dict(), save_path)
print("finished")