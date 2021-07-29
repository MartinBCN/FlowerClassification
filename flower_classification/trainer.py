from datetime import datetime
import numpy as np
import torch
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader
from datetime import datetime
import logging

from flower_classification.flower_classifier import FlowerClassifier

logger = logging.getLogger()


class FlowerTrainer(FlowerClassifier):

    def __init__(self):
        super().__init__()

    def evaluate(self, data_loader: DataLoader):

        predicted_labels = []
        ground_truth = []

        for i, (image, label) in enumerate(data_loader):

            with torch.no_grad():

                predicted_labels.append(self.model.predict(image).detach().cpu().numpy())
                ground_truth.append(label.detach().cpu().numpy())

        # Log accuracy
        predicted_labels = np.concatenate(predicted_labels)
        ground_truth = np.concatenate(ground_truth)
        accuracy = (ground_truth == predicted_labels).mean()
        return accuracy, predicted_labels, ground_truth

    def train(self, data_loader: dict, epochs: int, early_stop_epochs: int = 5):

        start = datetime.now()
        for epoch in range(epochs):
            logger.info(('=' * 125))
            logger.info(f'Epoch {epoch}')
            logger.info(('=' * 125))

            self.training_log['learning_rate'].append(self.get_lr())

            for phase in ['train', 'validation']:

                logger.info(('-' * 125))
                logger.info(f'Phase {phase}')
                logger.info(('-' * 125))

                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                if phase not in self.training_log.keys():
                    self.training_log[phase] = {'epoch_loss': [], 'batch_loss': []}

                epoch_loss = 0
                epoch_ground_truth = []
                epoch_predicted_labels = []
                batch_time = []

                for i, sample in enumerate(data_loader[phase]):
                    start_batch = datetime.now()
                    image = sample['image']
                    label = sample['keypoints']
                    if phase == 'train':
                        # zero the parameter gradients
                        self.optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = self.model(image)
                    loss = self.criterion(outputs, label.float())
                    epoch_loss += loss.detach().cpu().numpy()

                    # Batch Accuracy
                    batch_predicted_labels = outputs.detach().cpu().numpy().reshape(-1)
                    batch_true_labels = label.detach().cpu().numpy().reshape(-1)
                    epoch_predicted_labels.append(batch_predicted_labels)
                    epoch_ground_truth.append(batch_true_labels)

                    if phase == 'train':
                        loss.backward()
                        self.optimizer.step()

                    # Permanent Value Tracking
                    batch_loss = loss.detach().cpu().numpy()
                    self.training_log[phase]['batch_loss'].append(batch_loss)
                    accuracy = r2_score(batch_true_labels, batch_predicted_labels)
                    self.training_log[phase]['batch_accuracy'].append(accuracy)

                    # Regular Logging
                    log_string = f'Epoch {epoch + 1}/{epochs}, Batch {i+1}/{len(data_loader[phase])}'
                    log_string += f', Mean Epoch Loss: {epoch_loss / (i + 1):.2f}'
                    log_string += f'| Batch Loss: {batch_loss:.2f}'
                    current_accuracy = r2_score(np.concatenate(epoch_ground_truth),
                                                np.concatenate(epoch_predicted_labels))
                    log_string += f' |Epoch Accuracy: {current_accuracy:.2f}'
                    log_string += f'| Batch Accuracy: {accuracy:.2f}'
                    delta = (datetime.now() - start_batch).microseconds/image.shape[0]/1000
                    batch_time.append(delta)
                    log_string += f' | Time/image: {np.mean(batch_time):.2f}ms '

                    logger.info(log_string)

                # Log Epoch (accuracy and loss)
                accuracy = r2_score(np.concatenate(epoch_ground_truth), np.concatenate(epoch_predicted_labels))
                self.training_log[phase]['epoch_accuracy'].append(accuracy)
                self.training_log[phase]['epoch_loss'].append(epoch_loss)

            if self.scheduler is not None:
                self.scheduler.step()
