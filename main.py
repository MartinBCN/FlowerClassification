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


model = FlowerTrainer()
model.set_criterion('cross_entropy')
model.set_optimizer('adam', dict(lr=0.001))


loader = {phase: get_loader('data/flowers', phase) for phase in ['train', 'valid']}
