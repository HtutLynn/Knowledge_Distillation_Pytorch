"""
Implementation of the Teacher-Student Knowledge Distillation method
"""

# import the required models
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
# from torch.autograd import Variable
# Tensorboard functionality
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from customs import Functions, Metrics, progress_bar, DatasetGenerator
from tqdm import tqdm
import numpy as np
import time
import os
import copy
# from models.resnet import ResNet18

from models.vgg import VGG # student model

# Function for getting learning rate from optimizer
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train(model, optimizer, dataloader, temperature, alpha, epoch):
    """Train the model on batches
    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        temperature : (int) The value of temperature to be applied on the logits of model to gain softtarget
        alpha : Weight parameter to balance CrossEntropy loss and KL Divergence loss
        epoch: current epoch
    """

    # Set the model into train mode
    model.train()

    train_loss = 0
    KLD_loss = 0
    CE_loss = 0
    correct = 0
    total = 0
    datacount = len(dataloader)

    for batch_idx, (train_batch, labels_batch, logits_batch) in enumerate(dataloader):

        # move the data onto the device
        train_batch, labels_batch, logits_batch = train_batch.to(device), labels_batch.to(device), logits_batch.to(device)

        # # convert to torch Variables
        # train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)

        # squeeze the labels_batch to get rid of one dimension
        labels_batch = labels_batch.squeeze(1)
        # clear the previous grad 
        optimizer.zero_grad()

        # compute model outputs and loss
        outputs = model(train_batch)
        # loss = loss_fn_kd(outputs, labels_batch, logits_batch, temperature, alpha)
        loss, KLD, CE = all_kd_loss_fns(outputs, labels_batch, logits_batch, temperature, alpha)
        loss.backward()

        # after computing gradients based on current batch loss,
        # apply them to parameters
        optimizer.step()

        train_loss += loss.item()
        KLD_loss += KLD.item()
        CE_loss += CE.item()
        _, predicted = outputs.max(1)
        total += labels_batch.size(0)
        correct += predicted.eq(labels_batch).sum().item()

        # get learning rate
        current_lr = get_lr(optimizer=optimizer)

        # write to tensorboard
        writer.add_scalar('train/KD_loss', train_loss/(batch_idx+1), (datacount * (epoch+1)) + (batch_idx+1))
        writer.add_scalar('train/KL_Div', KLD_loss/(batch_idx+1), (datacount * (epoch+1)) + (batch_idx+1))
        writer.add_scalar('train/CE_loss', CE_loss/(batch_idx+1), (datacount * (epoch+1)) + (batch_idx+1))
        writer.add_scalar('train/accuracy', 100.*correct/total, (datacount * (epoch+1)) + (batch_idx+1))
        writer.add_scalar('Learning rate', current_lr)
        writer.add_scalar('Temperature', temperature)
        writer.add_scalar('Alpha', alpha)

        progress_bar(batch_idx, len(dataloader), 'Train Loss: %.3f | Train Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def eval(model, dataloader, temperature, alpha, epoch):
    """Evaluate the trained model's performance on Test data on batches
    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training datas
        temperature : (int) The value of temperature to be applied on the logits of model to gain softtarget
        alpha : Weight parameter to balance CrossEntropy loss and KL Divergence loss
        epoch: current epoch
    """

    # Set the model into test mode
    model.eval()

    test_loss = 0
    KLD_loss = 0
    CE_loss = 0
    correct = 0
    total = 0
    datacount = len(dataloader)

    # check global variable `best_accuracy`
    global best_accuracy

    with torch.no_grad():
        for batch_idx, (test_batch, labels_batch, logits_batch) in enumerate(dataloader):

            # move the data onto device
            test_batch, labels_batch, logits_batch = test_batch.to(device), labels_batch.to(device), logits_batch.to(device)

            # squeeze the labels_batch to get rid of one dimension
            labels_batch = labels_batch.squeeze(1)

            # compute the model output
            outputs = model(test_batch)
            # loss = loss_fn_kd(outputs, labels_batch, logits_batch, temperature, alpha)
            loss, KLD, CE = all_kd_loss_fns(outputs, labels_batch, logits_batch, temperature, alpha)
            
            test_loss += loss.item()
            KLD_loss += KLD.item()
            CE_loss += CE.item()
            _, predicted = outputs.max(1)
            total += labels_batch.size(0)
            correct += predicted.eq(labels_batch).sum().item()

            # log the test_loss
            writer.add_scalar('test/KD_loss', test_loss/(batch_idx+1), (datacount * (epoch+1)) + (batch_idx+1))
            writer.add_scalar('test/KL_Div', KLD_loss/(batch_idx+1), (datacount * (epoch+1)) + (batch_idx+1))
            writer.add_scalar('test/CE_loss', CE_loss/(batch_idx+1), (datacount * (epoch+1)) + (batch_idx+1))
            writer.add_scalar('test/accuracy', 100.*correct/total, (datacount * (epoch+1)) + (batch_idx+1))

            progress_bar(batch_idx, len(dataloader), 'Test Loss: %.3f | Test Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    current_loss = test_loss/len(dataloader)
    # save checkpoint
    acc = 100. * correct/total
    if acc > best_accuracy:
        print("Saving the model.....")
        save_path = "/home/htut/Desktop/Knowledge_Distillation_Pytorch/checkpoints/students/retake/VGG11_T4_a0.5_acc:{:.3f}_loss_{:.3f}.pt".format(acc, current_loss)
        torch.save(model.state_dict(), save_path)
        
        best_accuracy = acc

def train_and_evaluate(model, train_dataloader, test_dataloader, optimizer, scheduler, total_epochs, temperature, alpha):
    """Train the model and evaluate every epoch.
    Args:
        model: (torch.nn.Module) the neural network
        train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        test_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches validation data
        optimizer: (torch.optim) optimizer for parameters of model
        lr_scheduler: (torch.optim.lr_scheduler) Adjustment function for the learning rate
        loss_ft: a function that takes batch_output and batch_labels and computes the loss for the batch
        total_epochs: total number of epochs
    """

    for epoch in range(total_epochs):

        # Run one epoch for both train and test
        print("Epoch {}/{}".format(epoch + 1, total_epochs))

        # compute number of batches in one epoch(one full pass over the training set)
        train(model, optimizer, train_dataloader, temperature, alpha, epoch)

        scheduler.step()

        # Evaluate for one epoch on test set
        eval(model, optimizer, test_dataloader, temperature, alpha, epoch)

def loss_fn(outputs, labels):
    """
    Compute the cross entropy loss given outputs and labels.
    Args:
        outputs: (Variable) dimension batch_size x 6 - output of the model
        labels: (Variable) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]
    Returns:
        loss (Variable): cross entropy loss for all images in the batch
    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """
    return nn.CrossEntropyLoss()(outputs, labels)

# Knowledge Distillation loss (combined loss = KL divergence Loss + Cross Entropy Loss)
# Implementation is referenced from `https://github.com/peterliht/knowledge-distillation-pytorch`
def loss_fn_kd(outputs, labels, teacher_outputs, temperature, alpha):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities!
    """
    alpha = alpha
    T = temperature
    KD_loss = nn.KLDivLoss()(functional.log_softmax(outputs/T, dim=1),
                             functional.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
              functional.cross_entropy(outputs, labels) * (1. - alpha)

    return KD_loss

def all_kd_loss_fns(outputs, labels, teacher_outputs, temperature, alpha):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities!
    """
    alpha = alpha
    T = temperature
    KL_Divergence = nn.KLDivLoss()(functional.log_softmax(outputs/T, dim=1),
                             functional.softmax(teacher_outputs/T, dim=1))

    Cross_Entropy_loss = functional.cross_entropy(outputs, labels)

    KD_loss = KL_Divergence * (alpha * T * T) + Cross_Entropy_loss * (1. - alpha)

    return KD_loss, KL_Divergence, Cross_Entropy_loss



if __name__ == "__main__":
    

    F = Functions()
    M = Metrics()
    # The dataset that we are going to train the network is : CIFAR-10 dataset
    
    # Student model is PreAct-ResNet18 model
    # Therefore, logits are generated from PreAct-ResNet18 model
    train_logits = "/home/htut/Desktop/Knowledge_Distillation_Pytorch/KD_data/vgg/train_logits_vgg19.npy"
    test_logits = "/home/htut/Desktop/Knowledge_Distillation_Pytorch/KD_data/vgg/test_logits_vgg19.npy"

    # generate data for knowledge distillation
    kd_train_set, kd_test_set = DatasetGenerator.construct(train_logits=train_logits, test_logits=test_logits)

    trainloader = torch.utils.data.DataLoader(kd_train_set, batch_size=128,
                                            shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(kd_test_set, batch_size=100,
                                            shuffle=False, num_workers=4)

    # Setup hyperparameters
    temperature = 4
    alpha = 0.5

    classes = ('plane', 'car', 'bird', 'cat', 'deeer',
                'dog', 'frog', 'horse', 'ship', 'truck')

    # setup device for training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Setup best accuracy for comparing and model checkpoints
    best_accuracy = 0.0

    # setup Tensorboard file path
    writer = SummaryWriter('experiments/students/resnet/...')

    # Configure the Network
    # You can swap out any kind of architectire from /models in here
    # Student model is VGG11 architecture
    model_fn = VGG('VGG11')
    model_fn = model_fn.to(device)
    
    summary(model_fn, (3, 32, 32))

    # Setup the optimizer method for all the parameters
    # optimizer_fn = optim.SGD(model_fn.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    optimizer_fn = optim.SGD(model_fn.parameters(), lr=0.1, weight_decay=5e-4)

    scheduler = StepLR(optimizer_fn, step_size=50, gamma=0.1)

    train_and_evaluate(model=model_fn, train_dataloader=trainloader, test_dataloader=testloader,
                        optimizer=optimizer_fn, scheduler=scheduler, total_epochs=150, temperature=temperature, alpha=alpha)

    writer.close()