from typing import Dict

from PIL import Image
from matplotlib.figure import Figure
from torch import Tensor
import torch.nn as nn
from torchvision.transforms import transforms
import torch
import numpy as np
import matplotlib.pyplot as plt

from flower_classification.flower_training import FlowerTrainer


class FlowerInference(FlowerTrainer):
    """
    Inference for Flower Classification Model
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    def __init__(self, model_type: str, num_classes: int):
        super(FlowerInference, self).__init__(model_type=model_type, num_classes=num_classes)
        self.label_dictionary = None

    def set_label_dictionary(self, label_dictionary: Dict[int, str]) -> None:
        """
        Add a label dictionary

        Parameters
        ----------
        label_dictionary: Dict[int: str]

        Returns
        -------
        None
        """
        self.label_dictionary = label_dictionary

    def image_to_label(self, image: Image) -> str:
        """
        Calculate the most likely label for a Pillow image

        Parameters
        ----------
        image: Image

        Returns
        -------
        str
            Name of the classified flower
        """
        image = self.transform(image)
        return self.tensor_to_label(image)

    def tensor_to_label(self, image: Tensor) -> str:
        """
        Calculate the most likely label for a Tensor
        Parameters
        ----------
        image: Tensor

        Returns
        -------
        str
            Name of the classified flower
        """
        self.model.eval()
        image = image.to(self.device)
        image = image.unsqueeze(0)
        outputs = self.model(image)
        batch_predicted_labels = outputs.detach().cpu().numpy()
        batch_predicted_labels = np.argmax(batch_predicted_labels, axis=1)
        return batch_predicted_labels[0]

    def image_to_probability(self, image: Image, num_results: int = 5) -> Dict[str, float]:
        """
        Calculate the probabilities for the different labels and return the five most likely ones

        Parameters
        ----------
        image: Image
        num_results: int, default = 5

        Returns
        -------
        Dict[str, float]
            Dictionary of the form {name: probability}
        """
        image = self.transform(image)
        return self.tensor_to_probability(image, num_results)

    def tensor_to_probability(self, image: Tensor, num_results: int = 5) -> Dict[str, float]:
        """
        Calculate the probabilities for the different labels and return the five most likely ones

        Parameters
        ----------
        image: Tensor
        num_results: int, default = 5

        Returns
        -------
        Dict[str, float]
            Dictionary of the form {name: probability}
        """
        self.model.eval()
        image = image.to(self.device)
        image = image.unsqueeze(0)
        prediction = self.model(image)

        softmax = nn.Softmax(dim=1)
        prediction = softmax(prediction)

        top_predictions = torch.topk(prediction, num_results)
        top_indices = top_predictions.indices.detach().cpu().numpy()[0]
        top_values = top_predictions.values.detach().cpu().numpy()[0]

        if self.label_dictionary is None:
            prediction = {value: idx + 1 for idx, value in zip(top_indices, top_values)}
        else:
            prediction = {value: self.label_dictionary[f'{idx + 1}'] for idx, value in zip(top_indices, top_values)}
        return prediction

    def plot_topk(self, image: Image, true_label: int = None) -> Figure:
        """
        Create a Figure of the image together with a visualisation of the most likely predictions

        Parameters
        ----------
        image: Image
        true_label: int, default = None

        Returns
        -------
        Figure
        """
        predictions = self.image_to_probability(image)

        fig, axes = plt.subplots(1, 2, figsize=(20, 10))

        axes[0].imshow(image)

        # Example data
        names = tuple(predictions.values())
        y_pos = np.arange(len(names))
        probabilities = list(predictions.keys())

        axes[1].barh(y_pos, probabilities, align='center')
        axes[1].set_yticks(y_pos)
        axes[1].set_yticklabels(names)
        axes[1].invert_yaxis()  # labels read top-to-bottom
        axes[1].set_xlabel('Probability')

        if (true_label is not None) and (self.label_dictionary is not None):
            axes[0].set_title(f'True class: {self.label_dictionary[str(true_label)]}')

        return fig
