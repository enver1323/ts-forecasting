from argparse import ArgumentParser
from typing import Optional, Dict, Type
from generics import BaseTrainer, BaseConfig
from domain.point_id import PointIDConfig, PointIDTrainer
from domain.point_id_ar import PointIDARConfig, PointIDARTrainer
from dataclasses import is_dataclass, fields, asdict
from enum import Enum


class ParserConfigs(Enum):
    point_id = ('point_id', PointIDConfig, PointIDTrainer)
    point_id_ar = ('point_id_ar', PointIDARConfig, PointIDARTrainer)

    def __init__(self, key: str, config: Type[BaseConfig], trainer: Type[BaseTrainer]):
        self.key = key
        self.config = config
        self.trainer = trainer

class TrainParser(ArgumentParser):
    PARSER_CONFIG_KEYS: Dict[str, ParserConfigs] = {parser.key: parser for parser in ParserConfigs}

    def __init__(self, *args, **kwargs):
        super(TrainParser, self).__init__(*args, **kwargs)
        self.set_arg_rules()

    def set_arg_rules(self):
        subparsers = self.add_subparsers(title="", parser_class=ArgumentParser)
        for parser_config in ParserConfigs:
            name, config, _ = parser_config.value
            parser = subparsers.add_parser(name)
            parser.add_argument('--config', type=str, default=name)
            self._add_arguments_from_config(parser, config)

    def _add_arguments_from_config(self, parser: ArgumentParser, config, prefix: str = '--'):
        if not is_dataclass(config):
            return

        for field in fields(config):
            field_name = prefix + field.name
            if is_dataclass(field.type):
                self._add_arguments_from_config(
                    parser=parser,
                    config=field.default,
                    prefix=field_name + '.'
                )
                continue

            parser.add_argument(
                field_name,
                type=field.type,
                default=field.default
            )

    @staticmethod
    def parse_dict(config: dict) -> dict:
        result = {}

        for key, val in config.items():
            sub_keys = key.split('.')
            entry = result
            for sub_key in sub_keys[:-1]:
                if sub_key not in entry:
                    entry[sub_key] = {}
                entry = entry[sub_key]

            entry[sub_keys[-1]] = val

        return result

    def parse_args_to_dict(self, *args, **kwargs) -> dict:
        args = self.parse_args(*args, **kwargs)
        args_dict = vars(args)

        return self.parse_dict(args_dict)

    def parse_args_to_config(self, *args, **kwargs) -> Optional[BaseConfig]:
        src_data = self.parse_args_to_dict(*args, **kwargs)
        if src_data.get('config', None) not in self.PARSER_CONFIG_KEYS:
            return None

        config_cls = self.PARSER_CONFIG_KEYS[src_data['config']].config

        def _get_config(config, src_datum):
            config_args = {}
            if not is_dataclass(config):
                return None

            for field in fields(config):
                name = field.name
                if name not in src_datum or not(field.init):
                    continue

                config_args[name] = _get_config(field.type, src_datum[name]) \
                    if is_dataclass(field.type) else \
                    src_datum[name]

            return config(**config_args)

        return _get_config(config_cls, src_data)

    def get_trainer(self, *args, **kwargs) -> Optional[type[BaseTrainer]]:
        args = self.parse_args(*args, **kwargs)
        return self.PARSER_CONFIG_KEYS[args.config].trainer if args.config in self.PARSER_CONFIG_KEYS else None
