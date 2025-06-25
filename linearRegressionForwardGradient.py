import jax.numpy as jnp
from jax import grad, jit
import yaml
import matplotlib.pyplot as plt

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
    a = jnp.array(value_a, dtype=float)
    b = jnp.array(value_b, dtype=float)
    print(f'\nStarting training lReg with JAX...\n'
          f'\tYour Funktion: y = {a.item():.6f} * X + {b.item():.6f}\n'
          f'\tIterations: {iterations} \n '
          f'\tlearning rate: {learningRate} \n '
          f'\tprecision: {precision}')

    loss_list = []
    previous_digit = 0
    counter = 0
    compute_gradients = jit(grad(calculationMSE, argnums=(0,1)))
    for i in range(iterations):
        loss = calculationMSE(a, b)
        loss_list.append(loss.item())
        #print(f'Iteration: {i + 1} \n Loss: {loss.item():.4f} \n a: {a.item():.6f}\n b: {b.item():.6f}')
        # Update Parameters
        thet
        grad_a, grad_b = compute_gradients(a,b)
        a = a - learningRate * grad_a
        b = b - learningRate * grad_b
        digit = getDecimalDigit(a, precision)
        # print(f'Your New Funktion: y = {a.item():.6f} * X + {b.item():.6f}\n')

        # Meine "Converged?" Abfrage
        if digit == previous_digit:
            counter += 1
        else:
            counter = 0
        previous_digit = digit
        if counter >= patience:
            print(f"\tStopping early at Iteration {i + 1} as the {precision}-th decimal hasn´t changed")
            break

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
