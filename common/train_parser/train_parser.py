from argparse import ArgumentParser
from typing import Optional, Dict
from generics import BaseConfig
from dataclasses import is_dataclass, fields
from common.train_parser.config import PARSER_CONFIGS, TrainerType


class TrainParser(ArgumentParser):
    def __init__(self, *args, **kwargs):
        super(TrainParser, self).__init__(*args, **kwargs)
        self.set_arg_rules()

    def set_arg_rules(self):
        subparsers = self.add_subparsers(title="", parser_class=ArgumentParser)
        for parser_config in PARSER_CONFIGS.items():
            name, loader = parser_config
            config = loader.load_config()
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
        if src_data.get('config', None) not in PARSER_CONFIGS:
            return None

        config_cls = PARSER_CONFIGS[src_data['config']].load_config()

        def _get_config(config, src_datum):
            config_args = {}
            if not is_dataclass(config):
                return None

            for field in fields(config):
                name = field.name
                if name not in src_datum or not (field.init):
                    continue

                config_args[name] = _get_config(field.type, src_datum[name]) \
                    if is_dataclass(field.type) else \
                    src_datum[name]

            return config(**config_args)

        return _get_config(config_cls, src_data)

    def get_trainer(self, *args, **kwargs) -> Optional[TrainerType]:
        args = self.parse_args(*args, **kwargs)
        print(args)
        return PARSER_CONFIGS[args.config].load_trainer() if hasattr(args, "config") and args.config in PARSER_CONFIGS else None
