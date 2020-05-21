import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF
import random

import numpy as np
from PIL import Image
from tqdm import tqdm
import os
import os.path as ops

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

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
            tensor_image = sample["image"]

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