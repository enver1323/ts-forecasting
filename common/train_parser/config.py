from typing import Type, Tuple
from generics import BaseTrainer, BaseConfig
from enum import Enum
from importlib import import_module

TrainerType = Type[BaseTrainer]


class ConfigLoader():
    def __init__(self, config_module: Tuple[str, str], trainer_module: Tuple[str, str]):
        self.config_package, self.config_module = config_module
        self.trainer_package, self.trainer_module = trainer_module

    def load_config(self) -> Type[BaseConfig]:
        return getattr(import_module(self.config_package), self.config_module)

    def load_trainer(self) -> TrainerType:
        return getattr(import_module(self.trainer_package), self.trainer_module)


PARSER_CONFIGS = {
    'point_id': ConfigLoader(
        ('domain.point_id.config', 'PointIDConfig'),
        ('domain.point_id.trainer', 'PointIDTrainer')
    ),
    'point_id_ar': ConfigLoader(
        ('domain.point_id_ar.config', 'PointIDARConfig'),
        ('domain.point_id_ar.trainer', 'PointIDARTrainer')
    ),
    'lin_adapt': ConfigLoader(
        ('domain.lin_adapt.config', 'LinAdaptConfig'),
        ('domain.lin_adapt.trainer', 'LinAdaptTrainer')
    ),
    'point_predictor': ConfigLoader(
        ('domain.point_predictor.config', 'PointPredictorConfig'),
        ('domain.point_predictor.trainer', 'PointPredictorTrainer')
    ),
    'change_id': ConfigLoader(
        ('domain.change_id.config', 'ChangeIDConfig'),
        ('domain.change_id.trainer', 'ChangeIDTrainer')
    ),
    'ilinear': ConfigLoader(
        ('domain.ilinear.config', 'ILinearConfig'),
        ('domain.ilinear.trainer', 'ILinearTrainer')
    ),
    'mamba': ConfigLoader(
        ('domain.mamba.config', 'MambaConfig'),
        ('domain.mamba.trainer', 'MambaTrainer')
    )
}
