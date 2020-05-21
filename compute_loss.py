# import the required models
import torch
import torch.nn as nn
import torch.functional
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
# from torch.autograd import Variable

from customs import Functions, Metrics, progress_bar
from tqdm import tqdm
import numpy as np
import time
import os
import copy
# from models.resnet import ResNet18
from models.vgg import VGG

def eval(model, loss_fn, dataloader):
    """Evaluate the trained model's performance on Test data
    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training datas
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # Set the model into test mode
    model.eval()

    test_loss = 0
    correct = 0
    total = 0

    # check global variable `best_accuracy`
    global best_accuracy

    with torch.no_grad():
        for batch_idx, (test_batch, labels_batch) in enumerate(dataloader):

            # move the data onto device
            test_batch, labels_batch = test_batch.to(device), labels_batch.to(device)

            # compute the model output
            outputs = model(test_batch)
            loss = loss_fn(outputs, labels_batch)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)

            total += labels_batch.size(0)
            correct += predicted.eq(labels_batch).sum().item()

            progress_bar(batch_idx, len(dataloader), 'Test Loss: %.3f | Test Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    current_loss = test_loss/len(dataloader)
    
    print("Loss value of model on test data: {}".format(current_loss))


def compute_seperate_losses(model, loss_fn, dataloader):
    """ Compute losses for the data points that then model has mis-classified and correctly classified seperately
    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training datas
    """

    # Set the model into test mode
    model.eval()

    misclassified_loss = 0
    misclassified_total = 0

    correct_loss = 0
    correct_total = 0

    worst_loss = 0

    with torch.no_grad():
        for single_test_image, single_label in tqdm(dataloader):

            # move the data onto device
            single_test_image, single_label = single_test_image.to(device), single_label.to(device)

            # compute the model output
            outputs = model(single_test_image)
            loss = loss_fn(outputs, single_label)

            _, predicted = outputs.max(1)

            if predicted.eq(single_label).item():
                correct_total += 1

                # item() method extracts the loss’s value as a Python float.
                correct_loss += loss.item()
            else:
                misclassified_total += 1

                # item() method extracts the loss’s value as a Python float.
                misclassified_loss += loss.item()

                if misclassified_loss > worst_loss:
                    worst_loss = misclassified_loss
                    worst_image = single_test_image
                    true_class = single_label.item()
                    predicted_label = predicted.item()                

            
        mean_misclassified_loss = misclassified_loss/misclassified_total
        mean_correct_loss = correct_loss/correct_total

    print("Mean loss value of the data points that model has mis-classified : {:.3f}".format(mean_misclassified_loss))
    print("Total loss value of the data points that model has mis-classified : {:.3f}".format(misclassified_loss))
    print("Mean loss value of the data points that model has correctly classified : {:.3f}".format(mean_correct_loss))
    print("Total loss value of the data points that model has correctly classified : {:.3f}".format(correct_loss))

    # Show the worst image
    F.show_image(image_tensor=worst_image, mean=(0.4914, 0.4822, 0.4465),std=(0.2023, 0.1994, 0.2010), 
                true_class=true_class, predicted_class=predicted_label)


if __name__ == "__main__":
    

    F = Functions()
    M = Metrics()

    # weights_path = "checkpoints/teachers/vgg/VGG19_acc:93.28.pt"
    weights_path = "checkpoints/teachers/vgg/vgg11_dataaug_acc:91.95.pt"
    # weights_path = "checkpoints/students/vgg-vgg/VGG19_VGG11_T6_a0.5_acc:87.0.pt"
    # weights_path = "checkpoints/students/vgg-vgg/VGG19_VGG11_T4_a0.5_acc:86.56.pt"

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # The dataset that we are going to train the network is : CIFAR-10 dataset

    testset = torchvision.datasets.CIFAR10(root="/home/htut/Desktop/Knowledge_Distillation_Pytorch/datasets", train=False,
                                            download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                            shuffle=False, num_workers=4)
    
    singleloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                            shuffle=False, num_workers=0)

    classes = ('plane', 'car', 'bird', 'cat', 'deeer',
                'dog', 'frog', 'horse', 'ship', 'truck')

    # setup device for training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Configure the Network
    
    # You can swap out any kind of architectire from /models in here
    # model_fn = ResNet18()
    # model_fn = VGG('VGG11')
    model_fn = VGG('VGG11')
    model_fn = model_fn.to(device)


    # Load the model
    model_fn.load_state_dict(torch.load(weights_path))
    
    # Setup the loss function
    criterion = nn.CrossEntropyLoss()

    eval(model=model_fn, loss_fn=criterion, dataloader=testloader)

    compute_seperate_losses(model=model_fn, loss_fn=criterion, dataloader=singleloader)