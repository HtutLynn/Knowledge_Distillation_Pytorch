import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


import numpy as np
from customs import Functions, Metrics
# import the model
from models.resnet import ResNet18


transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='/home/htut/Desktop/Knowledge_Distillation_Pytorch/datasets', train=True,
                                        download=True, transform=transforms)
# change the batch size if GPU memory cannot handle it.
# This is not very elegant solution: Please refine it if you can.
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1000,
                                        shuffle=False, num_workers=4)

testset = torchvision.datasets.CIFAR10(root="/home/htut/Desktop/Knowledge_Distillation_Pytorch/datasets", train=False,
                                        download=True, transform=transforms)
testloader = torch.utils.data.DataLoader(testset, batch_size=1000,
                                        shuffle=False, num_workers=4)

# setup device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
weights_path = "/home/htut/Desktop/Knowledge_Distillation_Pytorch/checkpoints/resnet/resnet18_acc:87.51.pt"
train_logits_path = "/home/htut/Desktop/Knowledge_Distillation_Pytorch/KD_data/resnet/train_logits.npy"
test_logits_path = "/home/htut/Desktop/Knowledge_Distillation_Pytorch/KD_data/resnet/test_logits.npy"

model = ResNet18()
model.to(device)

# Load the model
model.load_state_dict(torch.load(weights_path))

# Set the model into evaluation mode
model.eval()

with torch.no_grad():
    all_train_logits = []
    all_test_logits = []
    for train_data in trainloader:
        train_images, _ = train_data
        train_images = train_images.to(device)
        train_logits = model(train_images)
        train_logits = train_logits.cpu().tolist()

        # extend current batch train_logits into all_train_logits list
        all_train_logits.extend(train_logits)
    
    for test_data in testloader:
        test_images, _ = test_data
        test_images = test_images.to(device)
        test_logits = model(test_images)
        test_logits = test_logits.cpu().tolist()

        # extend current batch test_logits into all_train_logits list
        all_test_logits.extend(test_logits)

# convert the lists into numpy array and convert them into float32 for data consistency
all_train_logits = np.array(all_train_logits).astype(np.float32)
all_test_logits = np.array(all_test_logits).astype(np.float32)


# save the logits as npy file
np.save(train_logits_path, all_train_logits)
np.save(test_logits_path, all_test_logits)