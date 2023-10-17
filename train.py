from common import TrainParser
import torch
import numpy as np
import os
from typing import Optional, Type, Tuple
from generics import BaseConfig, BaseTrainer

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="0.7"


def get_trainer_config() -> Tuple[Optional[BaseConfig], Optional[Type[BaseTrainer]]]:
    parser = TrainParser()
    config = parser.parse_args_to_config()
    trainer = parser.get_trainer()
    print("Loading app configuration ...")

    return config, trainer

def app():
    config, trainer_type = get_trainer_config()
    if config is None or trainer_type is None:
        return

    seed_val = config.seed
    torch.manual_seed(seed_val)
    np.random.seed(seed_val)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trainer = trainer_type(config, device)
    trainer.log.start(project="time_series_forecasting",name=trainer.experiment_key)
    trainer.train()
    trainer.test()
    trainer.log.finish()


if __name__ == '__main__':
    app()
