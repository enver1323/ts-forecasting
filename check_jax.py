import jax
from jax import numpy as jnp

print(jax.devices())

a = jnp.ones((2))
a = a * 10

@jax.jit
def test(_a):
    return (_a * 20).mean()

test(a)
jax.grad(test)(a)