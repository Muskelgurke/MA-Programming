from Old.BP import linearRegressionJax as lR_Jax
from Old.FGD import linearRegressionForwardGradientJAX as lR_JaxFG, linearRegressionForwardGradientTorch

from Old.FGD import linearRegressionForwardGradientTorch as lR_np_FG
from Old.FGD import linearRegressionForwardGradientTorch_AutoGrad as lR_np_FG_ad
import yaml

def startTraining():
    with open("Old/Old_config.yaml", "r") as file:
        config = yaml.safe_load(file)
    print(f'\nStarting training lReg with JAX...\n'
          f'Your Funktion: y = {config["a"]} * X + {config["b"]}\n'
          f'Settings:\n\tIterations: {config["iterations"]} \n '
          f'\tlearning rate: {config["learningRate"]} \n '
          f'\tprecision: {config["precision"]}')
    a_jax, b_jax = lR_Jax.startTrain()
    a_jax_fg, b_jax_fg = lR_JaxFG.startTrain()
    a_np_fg, b_np_fg = lR_np_FG.startTrain()
    a_np_fg_autoGrad, b_np_fg_autoGrad = lR_np_FG_ad.startTrain()

    # a_torch, b_torch = lR_Torch.startTrain()
    # print(f'Torch\t\t\t y = {a_torch:.6f} * X + {b_torch:.6f}')
    print(f'\nJAX - backprop\t\t\t\t\t y = {a_jax:.6f} * X + {b_jax:.6f}')
    print(f'JAX - FG\t\t\t\t\t\t y = {a_jax_fg:.6f} * X + {b_jax_fg:.6f}\n')
    print(f'Numpy - FG\t\t\t\t\t\t y = {a_np_fg:.6f} * X + {b_np_fg:.6f}\n')
    print(f'Numpy - FG with Auto Grad Lib\t y = {a_np_fg_autoGrad:.6f} * X + {b_np_fg_autoGrad:.6f}\n')



if __name__ == '__main__':
   startTraining()
