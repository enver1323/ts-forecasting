from jax import numpy as jnp
from jaxtyping import Array

def mae(prediction: Array, ground_truth: Array) -> Array:
    return (jnp.abs(prediction - ground_truth)).mean()

def mse(prediciton: Array, ground_truth: Array) -> Array:
    return ((prediciton - ground_truth) ** 2).mean()

def bce_loss(pred_y: Array, true_y: Array) -> Array:
    return -(true_y * jnp.log(pred_y) + (1 - true_y) * jnp.log(1 - pred_y)).sum()


def cosine_similarity(a: Array, b: Array, axis: int = -1) -> Array:
    a = a / jnp.linalg.norm(a, ord=2, axis=axis, keepdims=True)
    b = b / jnp.linalg.norm(b, ord=2, axis=axis, keepdims=True)
    return (a * b).sum(axis=axis)
