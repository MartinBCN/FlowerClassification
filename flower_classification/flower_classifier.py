from abc import ABC
from pathlib import Path
from typing import Union
import torch
from torch.nn import Module, L1Loss, MSELoss, SmoothL1Loss, CrossEntropyLoss
from torch.optim import Adam, SGD
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import numpy as np
import torchvision.models as models
import torch.nn as nn

plt.style.use('ggplot')


class FlowerClassifier:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    optimizers = {'adam': Adam, 'sgd': SGD}
    criterions = {'l1': L1Loss, 'smooth_l1': SmoothL1Loss, 'mse': MSELoss, 'cross_entropy': CrossEntropyLoss}
    scheduler_choices = {'steplr': StepLR}

    def __init__(self, num_classes: int = 100):

        self.model = self.get_model(num_classes)
        self.criterion = None
        self.optimizer = None
        self.__optimizer_choice = None
        self.scheduler = None
        self.__scheduler_choice = None
        self.scheduler_parameter = None
        self.hyper_parameter = {}

        self.training_log = {'train': {'batch_loss': [], 'epoch_loss': [], 'batch_accuracy': [], 'epoch_accuracy': []},
                             'validation': {'batch_loss': [], 'epoch_loss': [], 'batch_accuracy': [],
                                            'epoch_accuracy': []},
                             'learning_rate': []}

    def get_model(self, num_classes: int) -> Module:
        model_ft = models.resnet18(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        # Here the size of each output sample is set to 2.
        # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

        return model_ft.to(self.device)

    def set_criterion(self, criterion: str) -> None:
        self.criterion = self.criterions[criterion]()

    def set_optimizer(self, optimizer: str, hyper_parameter: dict = None) -> None:
        if hyper_parameter is None:
            hyper_parameter = {}
        self.optimizer = self.optimizers[optimizer](self.model.parameters(), **hyper_parameter)
        self.hyper_parameter = hyper_parameter
        self.__optimizer_choice = optimizer

    def set_scheduler(self, scheduler: str, params: dict):
        self.scheduler = self.scheduler_choices[scheduler](self.optimizer, **params)
        self.__scheduler_choice = scheduler
        self.scheduler_parameter = params

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    @classmethod
    def load(cls, filepath: Union[str, Path]):
        state = torch.load(filepath)

        new = cls()

        new.set_optimizer(state['optimizer'], state['hyper_parameter'])

        new.model.load_state_dict(state['model_state_dict'])
        new.optimizer.load_state_dict(state['optimizer_state_dict'])
        new.training_log = state['training_log']
        new.set_scheduler(state['scheduler'], state['scheduler_parameter'])

        return new

    def save(self, filepath: Union[str, Path]):

        state = {'model_state_dict': self.model.state_dict(),
                 'optimizer_state_dict': self.optimizer.state_dict(),
                 'training_log': self.training_log,
                 'optimizer': self.__optimizer_choice,
                 'hyper_parameter': self.hyper_parameter,
                 'scheduler': self.__scheduler_choice,
                 'scheduler_parameter': self.scheduler_parameter
                 }

        torch.save(state, filepath)

    def get_inference_model(self):
        return self.model

    def plot(self, filepath: Union[str, Path] = None):

        fig, axes = plt.subplots(2, 2, figsize=(20, 10))

        train_batch_loss = self.training_log['train']['batch_loss']
        axes[0, 0].plot(np.convolve(train_batch_loss, np.ones(10)/10, mode='same'), label='Train', color='red')
        axes[0, 0].plot(train_batch_loss, label='Train', color='red', alpha=0.2)
        val_batch_loss = self.training_log['validation']['batch_loss']
        axes[0, 0].plot(np.convolve(val_batch_loss, np.ones(10)/10, mode='same'), label='Train', color='blue')
        axes[0, 0].plot(val_batch_loss, label='Validation', color='blue', alpha=0.2)
        axes[0, 0].legend(title='Batch Loss')

        axes[0, 1].plot(self.training_log['train']['epoch_loss'], label='Train')
        axes[0, 1].plot(self.training_log['validation']['epoch_loss'], label='Validation')
        axes[0, 1].legend(title='Epoch Loss')

        train_batch_accuracy = self.training_log['train']['batch_accuracy']
        axes[1, 0].plot(np.convolve(train_batch_accuracy, np.ones(10)/10, mode='same'), label='Train', color='red')
        axes[1, 0].plot(train_batch_accuracy, label='Train', color='red', alpha=0.2)
        val_batch_accuracy = self.training_log['validation']['batch_accuracy']
        axes[1, 0].plot(np.convolve(val_batch_accuracy, np.ones(10)/10, mode='same'), label='Train', color='blue')
        axes[1, 0].plot(val_batch_accuracy, label='Validation', color='blue', alpha=0.2)
        axes[1, 0].legend(title='Batch accuracy')
        axes[1, 0].set_ylim(-0.2, 1.0)

        axes[1, 1].plot(self.training_log['train']['epoch_accuracy'], label='Train')
        axes[1, 1].plot(self.training_log['validation']['epoch_accuracy'], label='Validation')
        axes[1, 1].legend(title='Epoch accuracy')
        axes[1, 1].set_ylim(-0.2, 1.0)

        if filepath is None:
            plt.show()
        else:
            plt.savefig(filepath)
