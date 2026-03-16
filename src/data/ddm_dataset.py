import torch
from torch.utils.data import IterableDataset
import numpy as np
import os



class DDMDataset(IterableDataset):
    def __init__(self, generator_callable, **kwargs):
        self.generator = generator_callable
        self.save_samnples_flag = kwargs.get('save_samples', False)
    
    def __iter__(self):
        while True:
            data = next(self.generator()) 
            yield data
