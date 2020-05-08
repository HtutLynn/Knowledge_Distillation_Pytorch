import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.functional as F
import torchvision
import torchvision.transforms as transforms

from models.simplenet import simple_conv_net

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

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

dataiter = iter(testloader)

batch_images, labels_batch = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(batch_images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels_batch[j]] for j in range(4)))

# Construct an instance of the simple conv net

net = simple_conv_net()

model_path = "/home/htut/Desktop/Knowledge_Distillation_Pytorch/checkpoints/net/simple_net_0.1.pt"

print("Loading the trained model....")
net.load_state_dict(torch.load(model_path))
print("Model has been loaded!")

# setup the device to train with
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# place the model on GPU
net.to(device)
outputs = net(batch_images.to(device))
print(outputs)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))