import torch
import torch.nn as nn
import torch.nn.functional as F

logits = [[ 2.1549, -1.1920,  0.7175, -0.6373,  1.5231, -0.9695, -1.5344, -0.5611,
                1.2410, -0.2743],
          [ 2.6602,  4.1885, -2.9084, -1.3497, -2.2745, -2.9993, -3.0944, -3.0027,
                4.9739,  3.5433]]

tensor_logits = torch.Tensor(logits)

classes = ('plane', 'car', 'bird', 'cat', 'deeer',
            'dog', 'frog', 'horse', 'ship', 'truck')

outputs = F.softmax(tensor_logits, 1)
print(outputs)
value, predictions = torch.max(outputs, 1)
print(value)
print(predictions)
print(classes[predictions[0]])
print(classes[predictions[1]])

maxes = torch.max(tensor_logits, 1, keepdim=True)[0]
x_exp = torch.exp(tensor_logits - maxes)
x_exp_sum = torch.sum(x_exp, 1, keepdim=True)
output_custom = x_exp/x_exp_sum

test = output_custom[0]
print(torch.sum(test))

true_exp = torch.exp(tensor_logits)
true_exp_sum = torch.sum(true_exp, 1, keepdim=True)
true_outputs = true_exp / true_exp_sum

print(torch.allclose(outputs, true_outputs))
print(torch.allclose(outputs, output_custom))

t_logits = tensor_logits / 4
t_exp = torch.exp(t_logits)
t_exp_sum = torch.sum(t_exp, 1, keepdim=True)
t_outputs = t_exp / t_exp_sum
t_value, t_predictions = torch.max(t_outputs, 1)
print(t_outputs)
print(t_value)
print(t_predictions)
print(classes[t_predictions[0]])
print(classes[t_predictions[1]])