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

    def __init__(self, model_type: str, num_classes: int) -> None:

        self.model_type = model_type
        self.num_classes = num_classes
        self.model = self.get_model(model_type, num_classes)
        self.criterion = None
        self.__criterion_choice = None
        self.optimizer = None
        self.__optimizer_choice = None
        self.scheduler = None
        self.__scheduler_choice = None
        self.scheduler_parameter = None
        self.hyper_parameter = {}

        self.training_log = {'train': {'batch_loss': [], 'epoch_loss': [], 'batch_accuracy': [], 'epoch_accuracy': []},
                             'valid': {'batch_loss': [], 'epoch_loss': [], 'batch_accuracy': [], 'epoch_accuracy': []},
                             'test': {'batch_loss': [], 'epoch_loss': [], 'batch_accuracy': [], 'epoch_accuracy': []},
                             'learning_rate': []}

        self.label_dictionary = None

    def get_model(self, model_type: str, num_classes: int) -> nn.Module:
        """
        Get the model. For now this is fixed to the pre-trained ResNet18 with only the number of classes customizable.
        This should, however, be extended to allow several models

        Parameters
        ----------
        model_type: str
            Which pre-trained model to use as a base
        num_classes: int
            Number of classes

        Returns
        -------
        nn.Module
        """
        available_models = {'resnet50': self._get_resnet50}
        return available_models[model_type](num_classes)

    def _get_resnet50(self, num_classes: int):
        model_ft = models.resnet50(pretrained=True)
        for param in model_ft.parameters():
            param.requires_grad = False
        num_features = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_features, num_classes)

        return model_ft.to(self.device)

    def set_criterion(self, criterion: str = 'cross_entropy') -> None:
        self.criterion = self.criterions[criterion]()
        self.__criterion_choice = criterion

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
        if torch.cuda.is_available():
            state = torch.load(filepath)
        else:
            state = torch.load(filepath, map_location=torch.device('cpu'))

        num_classes = state['num_classes']
        model_type = state['model_type']

        new = cls(model_type=model_type, num_classes=num_classes)

        new.set_optimizer(state['optimizer'], state['hyper_parameter'])
        new.set_criterion(state['criterion'])

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

        state = {
            'num_classes': self.num_classes,
            'model_type': self.model_type,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_log': self.training_log,
            'optimizer': self.__optimizer_choice,
            'hyper_parameter': self.hyper_parameter,
            'scheduler': self.__scheduler_choice,
            'scheduler_parameter': self.scheduler_parameter,
            'criterion': self.__criterion_choice
                 }

        torch.save(state, filepath)
