import torch
import torch.nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import os
import os.path as ops
import sys
import time
import math

"""
Progress Bar mechanism is referenced from `https://github.com/kuangliu/pytorch-cifar`
"""

_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

class Functions(object):
    """
    Collection of custom functions such as Temperature layered softmax in pytorch 
    """
    def __init__(self):
        pass

    @staticmethod
    def temp_softmax(logits, temp):
        """
        Implementation of the Temperature layered softmax func mentioned in Hinton's KD paper\
        # This implementation is only for research purposes, in actual knowledge process, we are going to use
        softmax function from torch.nn.functional
        :params logits: Feature embeddings/Outputs of the neural network
        :params temp: Temperture value, to be used in softmax to get soft targets
        :return soft_targets: Resulted soft target distribution
        """
        if isinstance(logits, torch.Tensor):
            
            # check if logits is multi logits or single logits
            if len(logits.shape) > 1:
                # minus max value from respective outputs for numerical stability 
                maxes = torch.max(logits, 1, keepdim=True)[0]
                logits = logits - maxes
                temperatured_values = logits / temp
                # exponentiate all the values in the logits
                exp_values = torch.exp(temperatured_values)
                temperatured_exp_sum = torch.sum(exp_values, 1, keepdim=True)
            else:
                maxes = torch.max(logits)
                logits = logits - maxes
                temperatured_values = logits / temp
                exp_values = torch.exp(temperatured_values)
                temperatured_exp_sum = torch.sum(exp_values)

            soft_targets = exp_values / temperatured_exp_sum

            return soft_targets
        
        elif isinstance(logits, np.ndarray):
            # check if logits is multi logits or single logits
            if len(logits.shape) > 1:
                # minus max value from respective outputs for numerical stability 
                maxes = np.max(logits, axis=1)
                for i in range(logits.shape[0]):
                    logits[i] = logits[i] - maxes[i]

                temperatured_values = logits / temp
                # exponentiate all the values in the logits
                temperatured_exp_values = np.exp(temperatured_values)
                temperatured_exp_sum = np.sum(temperatured_values, axis=1)

                soft_targets = []
                for i in range(logits.shape[0]):
                    soft_targets.append(temperatured_exp_values[i] / temperatured_exp_sum[i])
            else:
                max = np.max(logits)
                logits = logits - max

                temperatured_values = logits / temp
                # exponentiate all the values in the logits
                temperatured_exp_values = np.exp(temperatured_values)
                temperatured_exp_sum = np.sum(temperatured_exp_values)

                soft_targets = temperatured_exp_values/temperatured_exp_sum
            
            return np.array(soft_targets)


    
    @staticmethod
    def draw_bar_chart(values, labels):
        """
        Draw a bar chart by using matplotlib library
        :params values: Values of labels, used in the y-axis of bar chart
        :params labels: Labels, to indicate what are the values, x-axis of the bar chart
        """
        indexes = np.arange(len(labels))
        plt.bar(indexes, values)

        # label the axes
        plt.xlabel('Classes', fontsize=10)
        plt.ylabel('Values', fontsize=10)
        plt.xticks(indexes, labels, fontsize=10)
        plt.title("Distribution of the Soft Targets")

        plt.show()

    def visualize_soft_targets(self, soft_target, classes):
        """
        Visualize the soft_targets in bar chart
        Only work on single targets
        :params soft_targets: Values outputed from temperatured_softmax func by using logits as input
        :params classes: Output classes
        """
        numpy_classes = classes

        if isinstance(soft_target, torch.Tensor):

            numpy_soft_target = soft_target.numpy()

            if len(numpy_soft_target.shape) > 1:
                print("Error: This is 2D array!")
                raise ValueError("This function only works for single soft targets.")

            elif len(numpy_soft_target) != len(numpy_classes):
                raise ValueError("The dimensions of soft targets and classes do not match!")

            self.draw_bar_chart(numpy_soft_target, numpy_classes)

        elif isinstance(soft_target, np.ndarray):

            if len(soft_target.shape) > 1:
                print("Error: This is 2D array!")
                raise ValueError("This function only works for single soft targets.")

            elif len(soft_target) != len(numpy_classes):
                raise ValueError("The dimensions of soft targets and classes do not match!")


            self.draw_bar_chart(soft_target, numpy_classes)

        elif isinstance(soft_target, list):
            soft_target = np.array(soft_target)
            if len(soft_target.shape) > 1:
                print("Error: This is 2D array!")
                raise ValueError("This function only works for single soft targets.")

            elif len(soft_target) != len(numpy_classes):
                raise ValueError("The dimensions of soft targets and classes do not match!")


            self.draw_bar_chart(soft_target, numpy_classes)

    @staticmethod
    def compute_param_count(model):
        """
        Compute the total number of trainable parameters in the model
        :param model: Neural Network model, constructed in pytorch framework
        :return param_count: Total number of trainable parameters in a network
        """
        param_count = 0
        for param in model.parameters():
            if param.requires_grad:
                param_count += param.numel()
        
        return param_count

    @staticmethod
    def show_image(image_tensor, mean, std, true_class, predicted_class):
        """
        Un-normalize the image tensor and show the image
        image_tensor: normalized pytorch image tensor
        mean: [0, 1 ,2] tensor, parameter of the normalization method
        std : [0, 1, 2] tensor, parameter of the normalization method
        """
        true_image = image_tensor.new(*image_tensor.size())
        
        # perform de-normalization per channel
        true_image[:, 0, :, :] = image_tensor[:, 0, :, :] * std[0] + mean[0]
        true_image[:, 1, :, :] = image_tensor[:, 1, :, :] * std[1] + mean[1]
        true_image[:, 2, :, :] = image_tensor[:, 2, :, :] * std[2] + mean[2]

        np_image = true_image.squeeze().cpu().numpy()

        plt.imshow(np.transpose(np_image, (1, 2, 0)))
        # plt.imshow(np_image)

        plt.title("The image with the worst loss")
        plt.xlabel("True class : {}, predicted class : {}".format(true_class, predicted_class))

        plt.show()

        
class Metrics(object):
    """
    Collection of metrics to be performed on the model
    """
    def __init__(self):
        pass

    @staticmethod
    def accuracy(total, correct):
        """
        Implementaion of typical accuracy metric
        """

        return 100 * correct / total

class CustomDataset(Dataset):
      """
      Create a dataset that can produced data_augmented images
      """
      def __init__(self, logits, dataset, data_aug=False, normalization=True):
            """
            Initialize the parameters(logits) for creating a custom dataset
            :param logits: [train/test] Logits from a teacher model as numpy npy file
            :param dataset: [train/test] dataset of cifar-10 dataset (not dataloader)
            :param data_aug: If data augmentation is enabled or not
            """
            self._logits = logits
            self._dataset = dataset
            self._data_aug = data_aug
            self._normalization = normalization

            if not self._is_source_data_complete():
                  raise ValueError("Input datas are not complete"
                                    "Wrong file types or"
                                    "Files doesn't exit.")
            
            if normalization:
                  self.normalize_fn = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))

            # create Tensordatasets
            self._KD_data_list = self.construct()
      
      def _is_source_data_complete(self):

            """
            Check if source datas are complete
            """
            _, file_extension = os.path.splitext(self._logits)
            correct_file_type = file_extension == '.npy'
            file_exist = ops.exists(self._logits)

            return correct_file_type and file_exist
      
      def __len__(self):
            return len(self._KD_data_list)

      def __getitem__(self, idx):
            if torch.is_tensor(idx):
                  idx = idx.tolist()
            
            sample = {"image": self._KD_data_list[idx][0], "label": self._KD_data_list[idx][1], "logit" : self._KD_data_list[idx][2]}

            if self._data_aug:
                  sample = self.transform(sample)
            
            if self._normalization:
                  sample = self.normalize(sample)

            return sample

      def normalize(self, sample):
            """
            Perform channel-wise normalization
            """
            image = sample["image"]

            if torch.is_tensor(image):
                tensor_image = image
            else:
                tensor_image = TF.to_tensor(image)
    
            normalized_tensor_image = self.normalize_fn(tensor_image)

            # re-assign the image back to sample
            
            sample["image"] = normalized_tensor_image

            return sample

      def transform(self, sample):
            """
            Perform image augmentation functions on sample['image']
            """

            image = sample["image"]
            
            # Do Random Crop on image
            i, j, h, w  = transforms.RandomCrop.get_params(
                  image, output_size=(32, 32)
            )
            image = TF.crop(image, i, j, h, w)

            # Do random horizontal flip
            if random.random() > 0.5:
                  image = TF.hflip(image)

            # Transform to tensor
            image = TF.to_tensor(image)

            # re-assign the transformed back to sample
            sample["image"] = image

            return sample

      def construct(self):
            """
            Accept logits that are generated from a teacher model
            Construct a dataset for knowledge distillation
            :param train_logits: logits generated for the train set of cifar-10 by a teacher model
            :param test_logits: 
            """

            # Load logits (.npy) files
            logits = np.load(self._logits)
            # convert the data type to have data type consistency
            logits_tensor = logits.astype(np.float32)

            KD_data_list = []

            for data, logit in zip(self._dataset, logits):
                  # `image` is PIL.Image.Image data type
                  # `label` is int
                  image, label = data

                  # change labels and logits into numpy array
                  # Do not convert image to Tensor because image augmentation function
                  # only accepts PIL images
                  label = torch.tensor([label])
                  logit = torch.from_numpy(logit)
                  
                  # create a data sample
                  data_sample = [image, label, logit]

                  # A giant list with lists made of numpy arrays
                  KD_data_list.append(data_sample)

            # # convert the lists containg tensors into tensors
            # image_tensor = torch.stack(all_images)
            # label_tensor = torch.stack(all_labels)

            return KD_data_list