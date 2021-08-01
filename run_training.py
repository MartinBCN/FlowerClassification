from argparse import ArgumentParser
from datetime import datetime

from flower_classification.flower_data import get_loader
from flower_classification.flower_training import FlowerTrainer
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def get_argparser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('--model_type', default='resnet50', type=str, help='Base model for transfer learning')

    parser.add_argument('--criterion', default='cross_entropy', type=str, help='Criterion')
    parser.add_argument('--optimizer', default='adam', type=str, help='Optimizer')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Learning Rate')

    parser.add_argument('--epochs', default=10, type=int, help='Number of epochs')
    parser.add_argument('--early_stop_epochs', default=5, type=int,
                        help='Number of epochs without improvement on the '
                             'validation score after which training is terminated')

    parser.add_argument('--batch_size', default=8, type=int, help='Batch size')
    parser.add_argument('--use_fraction', default=None, type=float,
                        help='Use only a fraction of the data sets for fast execution, e.g. while debugging')

    # Names and directories
    parser.add_argument('--data_dir', default='flowers', type=str,
                        help='Data directory. Needs to contain the sub-folders train, test, valid')
    parser.add_argument('--model_dir', default='models', type=str, help='Model directory')
    parser.add_argument('--figure_dir', default='figures', type=str, help='Figure directory')

    return parser


def main(model_type: str, criterion: str, optimizer: str, learning_rate: float,
         batch_size: int, epochs: int, early_stop_epochs: int, use_fraction: float,
         data_dir: str, model_dir: str, figure_dir: str):
    model_name = f'FlowerClassifier_{model_type}_{datetime.now().replace(second=0, microsecond=0)}'
    model = FlowerTrainer(model_type=model_type, num_classes=102)
    model.set_criterion(criterion)
    model.set_optimizer(optimizer, dict(lr=learning_rate))

    train_loader = get_loader(data_dir, 'train', batch_size=batch_size, use_fraction=use_fraction)
    valid_loader = get_loader(data_dir, 'valid', batch_size=batch_size, use_fraction=use_fraction)

    model.train(train_loader=train_loader, validation_loader=valid_loader,
                epochs=epochs, early_stop_epochs=early_stop_epochs)

    fig = model.plot()
    fig.savefig(f'{figure_dir}/{model_name}.png')

    model.save(f'{model_dir}/{model_name}.ckpt')


if __name__ == '__main__':
    p = get_argparser()
    arguments = p.parse_args()
    main(model_type=arguments.model_type, criterion=arguments.criterion,
         optimizer=arguments.optimizer, learning_rate=arguments.learning_rate,
         batch_size=arguments.batch_size, epochs=arguments.epochs, early_stop_epochs=arguments.early_stop_epochs,
         use_fraction=arguments.use_fraction,
         data_dir=arguments.data_dir, model_dir=arguments.model_dir, figure_dir=arguments.figure_dir)
