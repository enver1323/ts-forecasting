from dataclasses import dataclass, asdict

@dataclass
class BaseConfig:
    seed: int = 1024
    wandb_log: int = 0

    def to_dict(self):
        return asdict(self)