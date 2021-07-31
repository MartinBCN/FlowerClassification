from pathlib import Path
from typing import Union
import torch
from torch.nn import L1Loss, MSELoss, SmoothL1Loss, CrossEntropyLoss, NLLLoss
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR
import torchvision.models as models
import torch.nn as nn


class FlowerClassifier:
    """
    Wrapper for the classification model, optimiser and criterion
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    optimizers = {'adam': Adam, 'sgd': SGD}
    criterions = {'l1': L1Loss, 'smooth_l1': SmoothL1Loss, 'mse': MSELoss,
                  'cross_entropy': CrossEntropyLoss, 'neg_log_likelihood': NLLLoss}
    scheduler_choices = {'steplr': StepLR}

    def __init__(self, num_classes: int = 102):

        self.model = self.get_model(num_classes)
        self.criterion = None
        self.optimizer = None
        self.__optimizer_choice = None
        self.scheduler = None
        self.__scheduler_choice = None
        self.scheduler_parameter = None
        self.hyper_parameter = {}

        self.training_log = {'train': {'batch_loss': [], 'epoch_loss': [], 'batch_accuracy': [], 'epoch_accuracy': []},
                             'valid': {'batch_loss': [], 'epoch_loss': [], 'batch_accuracy': [], 'epoch_accuracy': []},
                             'learning_rate': []}

        self.label_dictionary = None

    def get_model(self, num_classes: int) -> nn.Module:
        """
        Get the model. For now this is fixed to the pre-trained ResNet18 with only the number of classes customizable.
        This should, however, be extended to allow several models

        Parameters
        ----------
        num_classes: int
            Number of classes

        Returns
        -------
        nn.Module
        """
        model_ft = models.resnet18(pretrained=True)
        num_ftrs = model_ft.fc.in_features
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
        """
        Load a model from a given file path

        Parameters
        ----------
        filepath: Union[str, Path]

        Returns
        -------
        FlowerClassifier
        """
        state = torch.load(filepath)

        new = cls()

        new.set_optimizer(state['optimizer'], state['hyper_parameter'])

        new.model.load_state_dict(state['model_state_dict'])
        new.optimizer.load_state_dict(state['optimizer_state_dict'])
        new.training_log = state['training_log']

        scheduler = state['scheduler']
        if scheduler is not None:
            new.set_scheduler(scheduler, state['scheduler_parameter'])

        return new

    def save(self, filepath: Union[str, Path]) -> None:
        """
        Serialize and save a model. To allow resuming training also optimizer, hyper-parameter or lr scheduler are
        included

        Parameters
        ----------
        filepath: Union[str, Path]

        Returns
        -------
        None
        """

        state = {'model_state_dict': self.model.state_dict(),
                 'optimizer_state_dict': self.optimizer.state_dict(),
                 'training_log': self.training_log,
                 'optimizer': self.__optimizer_choice,
                 'hyper_parameter': self.hyper_parameter,
                 'scheduler': self.__scheduler_choice,
                 'scheduler_parameter': self.scheduler_parameter
                 }

        torch.save(state, filepath)
