import equinox as eqx
import os


class EarlyStopping:
    def __init__(self, patience: int, filename="checkpoint.pth"):
        self.valid_factor = None
        self.patience = patience
        self.n_fails = 0
        self.filename = filename

    def step(self, valid_factor, model, path: str) -> bool:
        if self.valid_factor is None or valid_factor < self.valid_factor:
            self.valid_factor = valid_factor
            self.n_fails = 0
            self.save_checkpoint(model, path)
            return True

        self.n_fails += 1
        print(f"Early Stopping counter: {self.n_fails} / {self.patience}")
        if self.n_fails >= self.patience:
            return False

        return True

    def save_checkpoint(self, model, path: str):
        if not os.path.exists(path):
            os.makedirs(path)
        filepath = os.path.join(path, self.filename)
        eqx.tree_serialise_leaves(filepath, model)

    def load_checkpoint(self, model, path: str):
        filepath = os.path.join(path, self.filename)
        model = eqx.tree_deserialise_leaves(filepath, model)

        return model
