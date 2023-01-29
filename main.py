import os

import mindspore as ms
from mindspore import context

from utils.trainer_utils import TrainConfig
from flearn.trainer.fedavg import FedAvg
from flearn.trainer.ifca import IFCA
from flearn.trainer.fesem import FeSEM
from flearn.trainer.fedgroup import FedGroup

# Modify these settings as you need
def main_fedavg():
    config = TrainConfig('fmnist', 'cnn', 'fedavg', 'ms')
    trainer = FedAvg(config)
    trainer.train()

def main_ifca():
    config = TrainConfig('mnist', 'mlp', 'ifca')
    config.trainer_config['dynamic'] = False # whether migrate clients
    trainer = IFCA(config)
    trainer.train()

def main_fesem():
    config = TrainConfig('mnist', 'mlp', 'fesem')
    config.trainer_config['dynamic'] = False
    trainer = FeSEM(config)
    trainer.train()

def main_flexcfl():
    config = TrainConfig('fmnist', 'cnn', 'fedgroup', 'ms')
    config.trainer_config['dynamic'] = True
    config.trainer_config['shift_type'] = "all"
    config.trainer_config['swap_p'] = 0.05
    trainer = FedGroup(config)
    trainer.train()

os.environ["CUDA_VISIBLE_DEVICES"]="0"
ms.set_context(mode=context.GRAPH_MODE, device_target="CPU")
main_flexcfl()
