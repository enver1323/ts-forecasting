from common import TrainParser
import torch
import numpy as np
import os
from argparse import Namespace
from typing import Optional, Type, Tuple
from generics import BaseConfig, BaseTrainer
import random

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="0.7"


def get_trainer_config(*args, **kwargs) -> Tuple[Optional[BaseConfig], Optional[Type[BaseTrainer]]]:
    parser = TrainParser()
    config = parser.parse_args_to_config(*args, **kwargs)
    trainer = parser.get_trainer(*args, **kwargs)
    # args = ['point_predictor']
    # config = parser.parse_args_to_config(args)
    # trainer = parser.get_trainer(args)
    print("Loading app configuration ...")

    return config, trainer

def app(*args, **kwargs):
    config, trainer_type = get_trainer_config(*args, **kwargs)
    if config is None or trainer_type is None:
        return

    seed_val = config.seed
    random.seed(seed_val)
    torch.manual_seed(seed_val)
    np.random.seed(seed_val)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.autograd.set_detect_anomaly(True)

    trainer = trainer_type(config, device)
    trainer.train()
    trainer.test()
    return trainer


if __name__ == '__main__':
    app()
