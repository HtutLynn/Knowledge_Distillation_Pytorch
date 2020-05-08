import numpy as np
import torch
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

testset = torchvision.datasets.CIFAR10(root="/home/htut/Desktop/Knowledge_Distillation_Pytorch/datasets", train=False,
                                        download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=4)

classes = ('plane', 'car', 'bird', 'cat', 'deeer',
            'dog', 'frog', 'horse', 'ship', 'truck')

# Construct an instance of the simple conv net

print("Length of the testloader : {}".format(len(testloader)))
print("Total images : {}".format(len(testloader) * 4))
net = simple_conv_net()

model_path = "/home/htut/Desktop/Knowledge_Distillation_Pytorch/checkpoints/net/simple_net_0.1.pt"

print("Loading the trained model....")
net.load_state_dict(torch.load(model_path))
print("Model has been loaded!")

for name, param in net.named_parameters():
    if param.requires_grad:
        print("{} : {}".format(name, param.data.shape))

# setup the device to train with
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# place the model on GPU
net.to(device)

correct = 0
total = 0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

with torch.no_grad():
    for data in testloader:
        test_images, labels_batch = data[0].to(device), data[1].to(device)
        outputs = net(test_images)
        _, predicted = torch.max(outputs.data, 1)
        c = (predicted == labels_batch).squeeze()
        total += labels_batch.size(0)
        correct += (predicted == labels_batch).sum().item()
        for i in range(4):
            label = labels_batch[0]
            class_correct[label] += c[i].item()
            class_total[label] += 1

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))