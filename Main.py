from BP import linearRegressionJax as lR_Jax
from FGD import linearRegressionForwardGradient as lR_JaxFG
import yaml

def startJAXTraining():
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    print(f'\nStarting training lReg with JAX...\n'
          f'Your Funktion: y = {config["a"]} * X + {config["b"]}\n'
          f'Settings:\n\tIterations: {config["iterations"]} \n '
          f'\tlearning rate: {config["learningRate"]} \n '
          f'\tprecision: {config["precision"]}')
    a_jax, b_jax = lR_Jax.startTrain()
    a_jax_fg, b_jax_fg = lR_JaxFG.startTrain()
    # a_torch, b_torch = lR_Torch.startTrain()
    # print(f'Torch\t\t\t y = {a_torch:.6f} * X + {b_torch:.6f}')
    print(f'\nJAX - backprop\t y = {a_jax:.6f} * X + {b_jax:.6f}')
    print(f'JAX - FG\t\t y = {a_jax_fg:.6f} * X + {b_jax_fg:.6f}\n')


if __name__ == '__main__':
   startJAXTraining()
