from BP import linearRegression_BP_JAX as lR_Jax
from FGD import linearRegression_FGD as lR_JaxFG, relu_Linear_FGD as lR_ReLU_JaxFG
import yaml

if __name__ == '__main__':
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    print(f'\nStarting training lReg with JAX...\n'
          f'Your Funktion: y = {config["a"]} * X + {config["b"]}\n'
          f'Settings:\n\tIterations: {config["iterations"]} \n '
          f'\tlearning rate: {config["learningRate"]} \n '
          f'\tprecision: {config["precision"]}')

    a_jax, b_jax = lR_Jax.startTrain()
    a_jax_fgd, b_jax_fgd = lR_JaxFG.startTrain()
    a_jax_fgd_lrRelu, b_jax_fgd_lrRelu = lR_ReLU_JaxFG.startTrain()
    # a_torch, b_torch = lR_Torch.startTrain()

    #print(f'Torch\t\t\t y = {a_torch:.6f} * X + {b_torch:.6f}')
    print(f'\nJAX - BP\t y = {a_jax:.6f} * X + {b_jax:.6f}')
    print(f'JAX - FGD\t\t y = {a_jax_fgd:.6f} * X + {b_jax_fgd:.6f}')
    print(f'JAX - FGD - LR + ReLU\t\t y = {a_jax_fgd_lrRelu:.6f} * X + {b_jax_fgd_lrRelu:.6f}\n')