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
x_jax = jnp.array(data_x)
y_jax = jnp.array(data_y)

a = jnp.array(value_a)
b = jnp.array(value_b)

def startTrain():
    loss_list = []
    previous_digit = 0
    counter = 0
    grad_loss = jit(grad(calculationMSE))

    for i in range(iterations):
        Y_pred = forward(x_jax)
        print(Y_pred)
        loss = calculationMSE(Y_pred, y_jax)
        loss_list.append(loss.item())

        # Update Parameters
        a = a - learningRate * grad_loss(Y_pred, y_jax)
        b = b - learningRate * grad_loss(Y_pred, y_jax)
        digit = getDecimalDigit(a[0], precision)

        if digit == previous_digit:
            counter += 1
        else:
            counter = 0
        previous_digit = digit
        if counter >= patience:
            print(f"Stopping early at Iteration {i + 1} as the {precision}-th decimal hasn´t changed")
            print(f'Iteration: {i + 1} \n Loss: {loss.item():.4f} \n a: {a[0].item():.6f}\n b: {b[0].item():.6f}')
            break

    # Plotting the loss
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(loss_list, 'r')
    plt.grid(True, color='y')
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Loss over iterations")

    # Plotting the final result
    plt.subplot(1, 2, 2)
    plt.scatter(x_jax, y_jax, color='blue', label='Target Points')
    print("Before Drawing:")
    x_sorted = jnp.sort(x_jax)
    y_sorted = forward(x_sorted)
    print(x_sorted)
    print(y_sorted)
    plt.plot(x_sorted, y_sorted, color='red', label='Learned Line')
    plt.grid(True, color='y')
    plt.legend()
    plt.title(f'Final: y = {a[0].item():.2f}x + {b[0].item():.2f}')

    plt.tight_layout()
    plt.show()
# defining the function for forward pass for prediction
def forward(x):
    return a * x + b

def calculationMSE(y_pred, y):
    return jnp.mean((y_pred - y) ** 2)

def getDecimalDigit(value,position):
    as_str = f'{value:.{position + 2}f}'
    return int(as_str.split(".")[1][position - 1])

