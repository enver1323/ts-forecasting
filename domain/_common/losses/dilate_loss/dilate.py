from typing import Tuple
import torch
from torch import Tensor
from domain._common.losses.softdtw_loss.soft_dtw_torch import SoftDTW, calc_distance_matrix
from domain._common.dilate_loss.path_soft_dtw import PathDTW


def dilate_loss(outputs: Tensor, targets: Tensor, alpha: float = 0.5, gamma: float = 0.1, normalize: bool = False) -> Tuple[Tensor, Tensor, Tensor]:
    # outputs, targets: shape (batch_size, N_output, n_channels)
    batch_size, N_output, n_channels = outputs.shape
    softdtw = SoftDTW(gamma=gamma, normalize=normalize)
    loss_shape = softdtw(outputs, targets)

    path_dtw = PathDTW(gamma=gamma, normalize=normalize)
    path = path_dtw(outputs, targets)

    omega = torch.arange(1, N_output + 1) \
        .reshape(1, N_output, 1) \
        .repeat(batch_size, 1, n_channels)

    omega = calc_distance_matrix(omega, omega).to(outputs.device)

    loss_temporal = torch.sum(path*omega) / (N_output*N_output * n_channels)
    loss = alpha*loss_shape + (1-alpha)*loss_temporal
    return loss, loss_shape, loss_temporal
