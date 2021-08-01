import copy
import numpy as np
import torch
from matplotlib.figure import Figure
from sklearn.metrics import accuracy_score
from torch import Tensor
from torch.utils.data import DataLoader
from datetime import datetime
import logging
from flower_classification.flower_classifier import FlowerClassifier
import matplotlib.pyplot as plt
plt.style.use('ggplot')
logger = logging.getLogger(__name__)


class FlowerTrainer(FlowerClassifier):
    """
    Wrapper for the training routine
    """

    def __init__(self, model_type: str, num_classes: int):
        super().__init__(model_type=model_type, num_classes=num_classes)

    def evaluate(self, data_loader: DataLoader):
        """
        Evaluate a data set
        Parameters
        ----------
        data_loader

        Returns
        -------
        accuracy: float
            Mean accuracy for the full data set
        predicted_labels: np.array
            Numpy array of the predicted labels
        ground_truth: np.array
            Numpy array of the ground truth from the data set
        """
        self.model.eval()

        predicted_labels = []
        ground_truth = []

        with torch.no_grad():

            for i, (image, label) in enumerate(data_loader):
                batch_loss, batch_predicted_labels, batch_true_labels = self.calculate_batch('test', image, label)

                predicted_labels.append(batch_predicted_labels)
                ground_truth.append(batch_true_labels)

        # Log accuracy
        predicted_labels = np.concatenate(predicted_labels)
        ground_truth = np.concatenate(ground_truth)
        accuracy = (ground_truth == predicted_labels).mean()
        return accuracy, predicted_labels, ground_truth

    def train(self, train_loader: DataLoader, validation_loader: DataLoader,
              epochs: int, early_stop_epochs: int = 5) -> None:
        """
        Main training loop:
        * iterates training data and calculates loss/backpropagation
        * iterates validation set to calculate losses and accuracy

        Parameters
        ----------
        train_loader: DataLoader
            Torch DataLoader with the training set
        validation_loader: DataLoader
            Torch DataLoader with the validation set
        epochs: int
            Number of epochs
        early_stop_epochs: int, default = 5
            If the validation accuracy does not improve over this many consecutive epochs training is aborted and
            the model with the best validation score is stored
        Returns
        -------
        None
        """

        data_loader = {'train': train_loader, 'valid': validation_loader}
        logger.info(f'Start training. '
                    f'images in training set: {len(train_loader.dataset)}, '
                    f'images in validation set: {len(validation_loader.dataset)}')

        best_score = 0.0
        no_improvement = 0
        best_model = copy.deepcopy(self.model)

        for epoch in range(epochs):

            self.training_log['learning_rate'].append(self.get_lr())
            execution_time = {}

            for phase in ['train', 'valid']:
                start = datetime.now()

                model_state = {"train": self.model.train, 'valid': self.model.eval}
                model_state[phase]()

                if phase not in self.training_log.keys():
                    self.training_log[phase] = {'epoch_loss': [], 'batch_loss': []}

                epoch_loss = 0
                epoch_ground_truth = []
                epoch_predicted_labels = []

                for (image, label) in data_loader[phase]:
                    batch_loss, batch_predicted_labels, batch_true_labels = self.calculate_batch(phase, image, label)

                    epoch_predicted_labels.append(batch_predicted_labels)
                    epoch_ground_truth.append(batch_true_labels)

                    epoch_loss += batch_loss

                # Log Epoch (accuracy and loss)
                accuracy = accuracy_score(np.concatenate(epoch_ground_truth), np.concatenate(epoch_predicted_labels))
                self.training_log[phase]['epoch_accuracy'].append(accuracy)
                self.training_log[phase]['epoch_loss'].append(epoch_loss / len(data_loader[phase]))

                execution_time[phase] = (datetime.now() - start).seconds

            # Regular Logging
            log_string = f'Epoch {epoch + 1}/{epochs}'

            log_string += f' | Train Loss: {self.training_log["train"]["epoch_loss"][-1]:.2f}'
            log_string += f', Validation Loss: {self.training_log["valid"]["epoch_loss"][-1]:.2f}'

            log_string += f' | Train Accuracy: {self.training_log["train"]["epoch_accuracy"][-1]:.2f}'
            log_string += f', Validation Accuracy: {self.training_log["valid"]["epoch_accuracy"][-1]:.2f}'

            log_string += f' | Time Train: {execution_time["train"]:.2f}s'
            log_string += f', Time Validation: {execution_time["valid"]:.2f}s'

            logger.info(log_string)

            # Check if training improved validation score
            if accuracy > best_score:
                best_score = accuracy
                best_model = copy.deepcopy(self.model)
                no_improvement = 0
            else:
                no_improvement += 1

            if self.scheduler is not None:
                self.scheduler.step()

            if no_improvement == early_stop_epochs:
                self.model = best_model
                logger.info(f'Training aborted after {early_stop_epochs} without improvement')
                break

    def calculate_batch(self, phase: str, image: Tensor, label: Tensor):
        """
        Calculate a single batch:
        * backpropagation in training
        * loss
        * accuracy score

        Parameters
        ----------
        phase: str
        image: Tensor
        label: Tensor

        Returns
        -------
        batch_loss: float
        batch_predicted_labels: np.array
        batch_true_labels: np.array
        """
        image = image.to(self.device)
        label = label.to(self.device)

        if phase == 'train':
            # zero the parameter gradients
            self.optimizer.zero_grad()

        # forward + backward + optimize
        outputs = self.model(image)
        loss = self.criterion(outputs, label)

        # --- Batch Accuracy ---
        batch_predicted_labels = outputs.detach().cpu().numpy()
        batch_predicted_labels = np.argmax(batch_predicted_labels, axis=1)
        batch_true_labels = label.detach().cpu().numpy().reshape(-1)

        if phase == 'train':
            loss.backward()
            self.optimizer.step()

        # Permanent Value Tracking
        batch_loss = loss.detach().cpu().numpy()
        self.training_log[phase]['batch_loss'].append(batch_loss)
        accuracy = accuracy_score(batch_true_labels, batch_predicted_labels)
        self.training_log[phase]['batch_accuracy'].append(accuracy)
        return batch_loss, batch_predicted_labels, batch_true_labels

    def plot(self) -> Figure:
        """
        Create a 2x2 Matplotlib plot of the losses and accuracies

        Returns
        -------
        Figure
        """

        fig, axes = plt.subplots(2, 2, figsize=(20, 10))

        # --- [0, 0] Training Batches ---
        train_batch_loss = self.training_log['train']['batch_loss']
        lns1 = axes[0, 0].plot(np.convolve(train_batch_loss, np.ones(10)/10, mode='same'), label='Loss', color='red')
        axes[0, 0].plot(train_batch_loss, label=None, color='red', alpha=0.2)
        ax00 = axes[0, 0].twinx()
        train_batch_accuracy = self.training_log['train']['batch_accuracy']
        lns2 = ax00.plot(np.convolve(train_batch_accuracy, np.ones(10)/10, mode='same'), label='Accuracy', color='blue')
        ax00.plot(train_batch_accuracy, label='Accuracy', color='blue', alpha=0.2)
        lines = lns1 + lns2
        labels = [line.get_label() for line in lines]
        axes[0, 0].legend(lines, labels, title='Training Batches')

        # --- [0, 1] Epoch Loss ---
        axes[0, 1].plot(self.training_log['train']['epoch_loss'], label='Train')
        axes[0, 1].plot(self.training_log['valid']['epoch_loss'], label='Validation')
        axes[0, 1].legend(title='Epoch Loss')

        # --- [1, 0] Validation Batches ---
        valid_batch_loss = self.training_log['valid']['batch_loss']
        lns1 = axes[1, 0].plot(np.convolve(valid_batch_loss, np.ones(10)/10, mode='same'), label='Loss', color='red')
        axes[1, 0].plot(valid_batch_loss, label=None, color='red', alpha=0.2)
        ax10 = axes[1, 0].twinx()
        valid_batch_accuracy = self.training_log['valid']['batch_accuracy']
        lns2 = ax10.plot(np.convolve(valid_batch_accuracy, np.ones(10)/10, mode='same'), label='Accuracy', color='blue')
        ax10.plot(valid_batch_accuracy, label='Accuracy', color='blue', alpha=0.2)
        lines = lns1 + lns2
        labels = [line.get_label() for line in lines]
        axes[1, 0].legend(lines, labels, title='Validation Batches')

        # --- [1, 1] Epoch Accuracy ---
        axes[1, 1].plot(self.training_log['train']['epoch_accuracy'], label='Train')
        axes[1, 1].plot(self.training_log['valid']['epoch_accuracy'], label='Validation')
        axes[1, 1].legend(title='Epoch accuracy')
        axes[1, 1].set_ylim(0.0, 1.0)

        return fig
