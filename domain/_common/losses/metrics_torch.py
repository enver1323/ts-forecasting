from torch import Tensor


def kl_divergence(prediction: Tensor, ground_truth: Tensor) -> Tensor:
    return -(ground_truth * prediction.log() + (1 - ground_truth) * (1 - prediction.log())).mean()


def mse(prediciton: Tensor, ground_truth: Tensor) -> Tensor:
    return ((prediciton - ground_truth) ** 2).mean()


def mae(prediction: Tensor, ground_truth: Tensor) -> Tensor:
    return ((prediction - ground_truth).abs()).mean()


def lag_loss(prediction: Tensor, dim: int = -1) -> Tensor:
    start_slices = [slice(None)] * prediction.ndim
    start_slices[dim] = slice(1, None)
    end_slices = [slice(None)] * prediction.ndim
    end_slices[dim] = slice(None, -1)

    diff = prediction[start_slices] - prediction[end_slices]
    diff = diff[start_slices] - diff[end_slices]

    return diff.abs().mean()


