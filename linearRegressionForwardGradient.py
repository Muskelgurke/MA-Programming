import jax
import jax.numpy as jnp
from jax import random
import yaml
import matplotlib.pyplot as plt
from sympy.abc import theta
from sympy.printing.theanocode import theano

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

learningRate = config["learningRate"]
iterations = config["iterations"]
patience = config["patience"]
precision = 1e-5
value_a = config["a"]
value_b = config["b"]
data_x = config["x"]
data_y = config["y"]

X_jax = jnp.array(data_x)
ones = jnp.ones_like(X_jax)
X_jax = jnp.stack([X_jax,ones], axis=1)
print(X_jax)
Y_jax = jnp.array(data_y)


def linearFkt(a,b,x):
    return a * x + b

def calculationMSE(theta,X,y):
    Y_pred = X @ theta
    return jnp.mean((Y_pred - y) ** 2)

def startTrain(plot=False):
    print(f'\nStarting training lReg with JAX...\n'
          f'\tYour Funktion: y = {value_a} * X + {value_b}\n'
          f'\tIterations: {iterations} \n '
          f'\tlearning rate: {learningRate} \n '
          f'\tprecision: {precision}')
    loss_list = []
    best_loss = jnp.inf
    counter = 0
    theta = jnp.array([value_a,value_b], dtype=float)
    print(f'theta = {theta}')
    key = random.PRNGKey(42)

    for i in range(iterations):
        key, subkey = random.split(key)

        v = random.normal(key, shape=theta.shape)
        print(f'v = {v}')
        f_val, directional_derivative = jax.jvp(lambda th: calculationMSE(th,X_jax,Y_jax),(theta,),(v,))
        g_theta = directional_derivative * v
        print(f'g_theta = {g_theta}')
        theta = theta - learningRate * g_theta
        print(f'theta = {theta}')

        loss = f_val
        loss_list.append(loss.item())
        # print(f'Your New Funktion: y = {a.item():.6f} * X + {b.item():.6f}\n')

        # Early Stopping: Converged to?
        if jnp.abs(loss-best_loss) < precision:
            counter += 1
            if counter >= patience:
                print(f"\tStopping early at Iteration {i + 1}-Change < {precision}\n")
                break
        else:
            counter = 0
            best_loss = loss

    a_learned, b_learned = theta
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
        y_sorted = linearFkt(a_learned,b_learned,x_sorted)

        plt.plot(x_sorted, y_sorted, color='red', label='Learned Line')
        plt.grid(True, color='y')
        plt.legend()
        plt.title(f'Final: y = {a.item():.2f}x + {b.item():.2f}')

        plt.tight_layout()
        plt.show()


    return a_learned, b_learned

def getDecimalDigit(value,position):
    as_str = f'{value:.{position + 2}f}'
    return int(as_str.split(".")[1][position - 1])
