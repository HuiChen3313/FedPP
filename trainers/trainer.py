from functools import partial
import torch
import os
from tqdm import tqdm
import numpy as np

class Trainer:
    def __init__(self,
                 data,
                 model,
                 seq_length,
                 lr = 0.0001,
                 max_epoch = 100,
                 lr_scheduler = None,
                 optimizer = None,
                 experiment_name = 'TPP_experiment',
                 **kwargs
                 ):

        self.data = data
        self.model = model
        self.seq_length = seq_length
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)\
            if optimizer is None else optimizer

        self.lr_scheduler = lr_scheduler

        self.experiment_name = experiment_name

        self._max_epoch = max_epoch

    def train(self):
        message = "start training"
        # self._logger.info(message)

        for epoch in range(self._max_epoch):
            self._model = self._model.train()
