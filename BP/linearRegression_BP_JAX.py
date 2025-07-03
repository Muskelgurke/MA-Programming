import jax.numpy as jnp
from jax import grad, jit
import yaml
import matplotlib.pyplot as plt
from prompt_toolkit.utils import to_int
from triton.language import dtype

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

learningRate = config["learningRate"]
iterations = config["iterations"]
patience = config["patience"]
precision = config["precision"]
value_a = config["a"]
value_b = config["b"]
data_x = config["x"]
data_y = config["y"]

X_jax = jnp.array(data_x)
Y_jax = jnp.array(data_y)

def linearFkt(a,b,x):
    return a * x + b

def calculationMSE(a, b):
    Y_pred = linearFkt(a,b,X_jax)
    return jnp.mean((Y_pred - Y_jax) ** 2)



def startTrain(plot=False):
    print('\nstarting training linearRegressionJax - Backward Propagation...')
    a = jnp.array(value_a, dtype=float)
    b = jnp.array(value_b, dtype=float)
    loss_list = []
    counter = 0
    best_loss = jnp.inf
    save_grad = jnp.zeros((2, iterations), dtype=float)
    compute_gradients = jit(grad(calculationMSE, argnums=(0,1)))

    for i in range(iterations):
        loss = calculationMSE(a, b)
        loss_list.append(loss.item())
        #print(f'Iteration: {i + 1} \n Loss: {loss.item():.4f} \n a: {a.item():.6f}\n b: {b.item():.6f}')
        # Update Parameters
        grad_a, grad_b = compute_gradients(a,b)
        save_grad = save_grad.at[:,i].set([grad_a,grad_b])
        print(f'grad_a: {grad_a.item():.6f}, grad_b: {grad_b.item():.6f}')
        a = a - learningRate * grad_a
        b = b - learningRate * grad_b

        if jnp.abs(loss-best_loss) < float(precision):
            counter += 1
            if counter >= patience:
                print(f"\n\tStopping early at Iteration {i + 1}-Change < {precision}\n")
                break
        else:
            counter = 0
            best_loss = loss


    # Plotting the loss
    if plot:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(loss_list, 'r')
        plt.grid(True, color='y')
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.title("Loss over iterations")

        # Plotting the final result
        plt.subplot(1, 2, 2)
        plt.scatter(X_jax, Y_jax, color='blue', label='Target Points')

        x_sorted = jnp.sort(X_jax)
        y_sorted = linearFkt(a,b,x_sorted)

        plt.plot(x_sorted, y_sorted, color='red', label='Learned Line')
        plt.grid(True, color='y')
        plt.legend()
        plt.title(f'Final: y = {a.item():.2f}x + {b.item():.2f}')

        plt.tight_layout()
        plt.show()

    return a.item(),b.item()

def getDecimalDigit(value,position):
    as_str = f'{value:.{position + 2}f}'
    return int(as_str.split(".")[1][position - 1])
