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


def main():
    model = FlowerTrainer(num_classes=102)
    model.set_criterion('cross_entropy')
    model.set_optimizer('adam', dict(lr=0.001))

    batch_size = 8
    use_fraction = None
    train_loader = get_loader('data/flowers', 'train', batch_size=batch_size, use_fraction=use_fraction)
    valid_loader = get_loader('data/flowers', 'valid', batch_size=batch_size, use_fraction=use_fraction)

    model.train(train_loader=train_loader, validation_loader=valid_loader, epochs=10, early_stop_epochs=5)

    fig = model.plot()
    fig.savefig('figures/test.png')

    model.save('models/test.ckpt')


if __name__ == '__main__':
    main()
