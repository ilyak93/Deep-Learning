import math

import numpy as np
import torch
import torch.utils.data.sampler as sampler
from torch.utils.data import Dataset


def create_train_validation_loaders(dataset: Dataset, validation_ratio,
                                    batch_size=100, num_workers=2):
    """
    Splits a dataset into a train and validation set, returning a
    DataLoader for each.
    :param dataset: The dataset to split.
    :param validation_ratio: Ratio (in range 0,1) of the validation set size to
        total dataset size.
    :param batch_size: Batch size the loaders will return from each set.
    :param num_workers: Number of workers to pass to dataloader init.
    :return: A tuple of train and validation DataLoader instances.
    """
    if not(0.0 < validation_ratio < 1.0):
        raise ValueError(validation_ratio)

    # TODO:
    #  Create two DataLoader instances, dl_train and dl_valid.
    #  They should together represent a train/validation split of the given
    #  dataset. Make sure that:
    #  1. Validation set size is validation_ratio * total number of samples.
    #  2. No sample is in both datasets. You can select samples at random
    #     from the dataset.
    #  Hint: you can specify a Sampler class for the `DataLoader` instance
    #  you create.
    # ====== YOUR CODE: ======
    dataset_size = len(dataset)
    permutation = torch.randperm(dataset_size)
    valid_size = int(dataset_size * validation_ratio)
    train_size = dataset_size - valid_size
    
    valid_sampler = torch.utils.data.SubsetRandomSampler(permutation[:valid_size])
    train_sampler = torch.utils.data.SubsetRandomSampler(permutation[valid_size:])
    
    dl_valid = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size, num_workers=0,
        sampler=valid_sampler
    )
    dl_train = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size, num_workers=0,
        sampler=train_sampler
    )
    # ========================

    return dl_train, dl_valid

