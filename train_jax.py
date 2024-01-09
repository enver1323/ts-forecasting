from common import TrainParser
import torch
import numpy as np
import os
from argparse import Namespace
from jax import random as jrandom
from typing import Optional, Type, Tuple
from generics import BaseConfig, BaseTrainer

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="0.1"


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
    torch.manual_seed(seed_val)
    np.random.seed(seed_val)
    key = jrandom.PRNGKey(seed_val)

    trainer = trainer_type(config, key)
    print(trainer.experiment_key)
    print("Training model ...")
    trainer.train()
    print("Testing model ...")
    trainer.test()
    return trainer


if __name__ == '__main__':
    app()

