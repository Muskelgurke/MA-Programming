import jax.numpy as jnp

def selu(x,alpha=1.67, lmbda=1.05):
    return lmbda * jnp.where(x>0, x, alpha*jnp.exp(x)-alpha)

x = jnp.arange(-5,5) # array with 5 elments, 0 to 4
print(x)
print(selu(x))

