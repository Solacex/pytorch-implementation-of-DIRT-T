from torch.utils.data import Dataset

from torchvision import datasets, transforms

import h5py
import torch
from torch import nn

import torch.nn.functional as F

import torch.utils.data

class JointLoader:

    def __init__(self, *datasets, collate_fn = None):

        self.datasets  = datasets
        self.iterators = [None] * len(datasets)
        self.collate_fn = collate_fn

    def __len__(self):

        return min([len(d) for d in self.datasets])

    def __iter__(self):
        for i, dataset in enumerate(self.datasets):
            self.iterators[i] = dataset.__iter__()
        return self

    def __next__(self):
        try:
            items = []
            for dataset in self.iterators:
                items.append(dataset.__next__())
        except StopIteration:
            raise StopIteration

        if self.collate_fn is not None:
            items = self.collate_fn(items)

        return items

