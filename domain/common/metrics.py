from torch import Tensor


def kl_divergence(prediction: Tensor, ground_truth: Tensor) -> Tensor:
    return -(ground_truth * prediction.log() + (1 - ground_truth) * (1 - prediction.log())).mean()


def mse(prediciton: Tensor, ground_truth: Tensor) -> Tensor:
    return ((prediciton - ground_truth) ** 2).mean()


def mae(prediction: Tensor, ground_truth: Tensor) -> Tensor:
    return ((prediction - ground_truth).abs()).mean()
