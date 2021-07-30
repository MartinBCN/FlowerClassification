from flower_classification.flower_data import get_loader
from flower_classification.flower_training import FlowerTrainer
import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


model = FlowerTrainer(num_classes=102)
model.set_criterion('cross_entropy')
model.set_optimizer('adam', dict(lr=0.001))


loader = {phase: get_loader('data/flowers', phase, batch_size=8) for phase in ['train', 'valid']}

model.train(data_loader=loader, epochs=5, early_stop_epochs=5)
