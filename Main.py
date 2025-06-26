import linearRegressionJax as lR_Jax
import linearRegressionTorch as lR_Torch
import linearRegressionForwardGradient as lR_JaxFG

if __name__ == '__main__':
    a_jax, b_jax = lR_Jax.startTrain()
    a_torch, b_torch = lR_Torch.startTrain()
    a_jax_fg, b_jax_fg = lR_JaxFG.startTrain()

    print(f'\nJAX - backprop\t y = {a_jax:.6f} * X + {b_jax:.6f}')
    print(f'Torch\t y = {a_torch:.6f} * X + {b_torch:.6f}\n')
    print(f'JAX - FG\t y = {a_jax_fg:.6f} * X + {b_jax_fg:.6f}\n')