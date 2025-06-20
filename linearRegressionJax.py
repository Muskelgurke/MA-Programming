import jax.numpy as jnp
from jax import grad, jit
import numpy as np
import matplotlib.pyplot as plt

# defining the function for forward pass for prediction
def forward(x):
    return a * x + b

def calculationMSE(y_pred, y):
    return jnp.mean((y_pred - y) ** 2)

def getDecimalDigit(value,position):
    as_str = f'{value:.{position + 2}f}'
    return int(as_str.split(".")[1][position - 1])

# Generate some sample data
x_jax = jnp.array([1,2,-1,1])
y_jax = jnp.array([2,3,-1,-1])

a = jnp.array([1.0])
b = jnp.array([2.0])

learningRate = 0.01
loss_list = []
iterations = 5000
previous_digit = 0
patience = 5
counter = 0
precision = 5
grad_loss = jit(grad(calculationMSE))
for i in range(iterations):
    Y_pred = forward(x_jax)
    loss = calculationMSE(Y_pred, y_jax)
    loss_list.append(loss.item())


    # Update Parameters
    a = a - learningRate * grad_loss(Y_pred, y_jax)
    b = b - learningRate * grad_loss(Y_pred, y_jax)
    digit = getDecimalDigit(a[0], precision)

    print(f'Iteration: {i} \n Loss: {loss.item():.4f} \n a: {a[0].item():.6f}\n b: {b[0].item():.6f}')
    if digit == previous_digit:
        counter +=1
    else:
        counter = 0
    previous_digit = digit
    if counter >= patience:
        print(f"Stopping early at epoch {i} as the {precision}-th decimal hasn´t changed")
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
plt.plot(x_jax, forward(x_jax), color='red', label='Learned Line')
plt.grid(True, color='y')
plt.legend()
plt.title(f'Final: y = {a[0].item():.2f}x + {b[0].item():.2f}')

plt.tight_layout()
plt.show()
