import torch
import torch.nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import os
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
        Implementation of the Temperature layered softmax func mentioned in Hinton's KD paper
        :params logits: Feature embeddings/Outputs of the neural network
        :params temp: Temperture value, to be used in softmax to get soft targets
        :return soft_targets: Resulted soft target distribution
        """

        temperatured_values = logits / temp
        # exponentiate all the values in the logits
        exp_values = torch.exp(temperatured_values)
        temperatured_exp_sum = torch.sum(exp_values, 1, keepdim=True)
        soft_targets = exp_values / temperatured_exp_sum

        return soft_targets
    
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
