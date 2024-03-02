"""
Containts functionality for creating Pytorch DataLoaders 
for image classification data.
"""

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import os

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
    train_dir: str,
    test_dir: str,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int=NUM_WORKERS
  ):
  """Creates training and testing DataLoaders.
  
  Takes in a training directory and testing directory path and turns
  them into Pytorch Datasets and then into Pytorch DataLoaders.

  Args:
    train_dir: Path to training directory.
    test_dir: Path to testing directory.
    transform: torchvision transform to perform on training and testing data.
    batch_size: Number of sample per batch in each of the DataLoaders.
    num_workers: An integer fo number of workers epr DataLoaders.

  Returns: 
    A tupple of (train_dataloader, test_dataloader, classnames).
    Where class_names is a list of the target classes.
    Example usage:
    train_dataloader, test_dataloader, classnames = \
      create_dataloaders (train_dir=path/to/train_dir,
                          test_dir=path/to/test_dir,
                          transform=some_transform,
                          batch_size=32,
                          num_workers=4)
  """
  # use ImageFolder to create dataset(s)
  train_data = datasets.ImageFolder(train_dir, transform=transform)
  test_data = datasets.ImageFolder(test_dir, transform=transform)

  # get classnames
  classnames = train_data.classes

  # turn image into data loaders
  train_dataloader = DataLoader(dataset=train_data,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=num_workers,
                                pin_memory=True)
  test_dataloader = DataLoader(dataset=test_data,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers,
                                pin_memory=True)
  
  return train_dataloader, test_dataloader, classnames

