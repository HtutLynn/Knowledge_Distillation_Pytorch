# Knowledge Distillation Pytorch
This is a repository for experimenting knowledge distillation methods.
The idea is mainly based on the paper
["Distilling the Knowledge in a Neural Network"](https://arxiv.org/abs/1503.02531) by Geoffrey Hinton, Oriol Vinyals, Jeff Dean.

The repo is entirely implemented in [pytorch](https://pytorch.org/) framework and consists of knowledge distillation processes and pipelines.

### dependencies
```sh
python 3.6
torch
torchvision
torchsummary
tensorboard
numpy
matplotlib
tqdm
```

### Installation

This repo requires [anaconda](https://www.anaconda.com/) environments to simulate the experiments.

Clone the repo and install the dependencies.

```sh
$ git clone git@github.com:HtutLynn/Knowledge_Distillation_Pytorch.git
$ conda create --name kd_pytorch
$ conda activate kd_pytorch
$ pip install -r requirements.txt
```

### Datasets

* [x] cifar-10
* [ ] mini open images

### Performance of models on cifar-10

| __Architectures__ | __resnet18__ | __preact resnet18__ | __googlenet__ | __densenet__ | __vgg16__  | __mobilenet__ |
|-------------------|--------------|---------------------|---------------|--------------|------------|---------------|
| Epochs            | 350          | 350                 | 350           | 350          | 150        | 350           |
| Loss Function     | CE           | CE                  | CE            | CE           | CE         | CE            |
| train loss        | 0.004        | 0.001               | N/A           | N/A          | 0.002      | N/A           |
| train accuracy    | 99.92        | 100.00              | N/A           | N/A          | 99.97      | N/A           |
| test loss         | 0.582        | 0.206               | N/A           | N/A          | 0.314      | N/A           |
| test accuracy     | 87.23        | 94.79               | N/A           | N/A          | 93.170     | N/A           |
| time taken        | 256m         | 259m                | N/A           | N/A          | 65m        | N/A           |

### Experiments

[Experiment #0](https://docs.google.com/document/d/12i2oMNmtcywhCxVUIJyn9bYolVUK8BDUoRc3xWFRYFU/edit?usp=sharing)

[Experiment #1](https://docs.google.com/document/d/1tF1t13Ht0psSqV3v6T8xUzIXuu30N1oU-A9p2QUna2U/edit?usp=sharing)

### Todos

 * [x] Data augmentation performable torch.utils.data.TensorDataset dataset
 * [x] Tensorboard Visualizations
 * [x] Compute_loss function for computing the mean loss for the data points that the model has mis-classified and correctly classified for analyisis purpose.
 * [ ] Multi-teachers => Multi-heads Knowledge distillation
 * [ ] Perfomance metrics
 * [ ] Google Colab trainable/runnable notebooks
 * [ ] Refactor the codebase to more readable format/style
 * [ ] Finding the best hyparameters for knowledge distillation methods


### References
[https://github.com/peterliht/knowledge-distillation-pytorch](https://github.com/peterliht/knowledge-distillation-pytorch)


[Pytorch-cifar](https://github.com/kuangliu/pytorch-cifar)
