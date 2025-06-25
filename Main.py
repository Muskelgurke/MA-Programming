import linearRegressionJax as lR_Jax
import linearRegressionTorch as lR_Torch

if __name__ == '__main__':
    a_jax, b_jax = lR_Jax.startTrain()
    a_torch, b_torch = lR_Torch.startTrain()

    print(f'\nJAX\t\t y = {a_jax:.6f} * X + {b_jax:.6f}')
    print(f'Torch\t y = {a_torch:.6f} * X + {b_torch:.6f}\n')
