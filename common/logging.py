from typing import Dict, TextIO
from collections import defaultdict


class TrainerStat():
    def __init__(self, stat_key: str, stat_type=float):
        self.stat_key = stat_key
        self.stat_type = stat_type
        self.reset()

    def reset(self):
        self.stat = defaultdict(self.stat_type)

    def set_vals(self, stat: dict):
        self.stat = stat

    def add_vals(self, stat: dict):
        for title, datum in stat.items():
            self.stat[f"{self.stat_key}/{title}"] += datum

    def scale_vals(self, factor: float):
        for title, datum in self.stat.items():
            self.stat[title] = datum / factor

    def format_to_str(self):
        return ", ".join([f"{title}: {val}" for title, val in self.stat.items()])

    def write_text(self, file: TextIO):
        file.write(self.format_to_str())


class HasLogging():
    def __init__(self, stat_splits=['train', 'valid', 'test']):
        self.stats: Dict[str, TrainerStat] = {}
        for stat_split in stat_splits:
            self.stats[stat_split] = TrainerStat(stat_split)

    def _stat_split_exists(self, stat_split):
        return stat_split in self.stats

    def add_stat(self, stat_split: str, stat: dict):
        if self._stat_split_exists(stat_split):
            self.stats[stat_split].add_vals(stat)

    def scale_stat(self, stat_split: str, factor: float):
        if self._stat_split_exists(stat_split):
            self.stats[stat_split].scale_vals(factor)

    def get_stat(self, stat_split: str, stat_name: str):
        if not self._stat_split_exists(stat_split):
            return None
        stat = self.stats[stat_split].stat
        return stat.get(stat_name, None)

    def reset_stat(self, stat_split: str):
        if self._stat_split_exists(stat_split):
            self.stats[stat_split].reset()

    def show_stat(self, stat_split: str):
        if self._stat_split_exists(stat_split):
            print(self.stats[stat_split].format_to_str())

    def write_text(self, stat_split: str, file: TextIO):
        if self._stat_split_exists(stat_split):
            self.stats[stat_split].write_text(file)
            file.write('\n')

    def write_text_all(self, file: TextIO):
        for stat_split in self.stats:
            self.stats[stat_split].write_text(file)
            file.write('\n')
