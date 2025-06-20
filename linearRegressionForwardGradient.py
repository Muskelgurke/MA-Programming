import torch
import numpy as np
import matplotlib.pyplot as plt


# defining the function for forward pass for prediction
def forward(x):
    return a * x + b


def calculationMSE(y_pred, y):
    return torch.mean((y_pred - y) ** 2)

def getDecimalDigit(value,position):
    as_str = f'{value:.{position + 2}f}'
    return int(as_str.split(".")[1][position - 1])

X = torch.tensor([1,2,-1,1])
Y = torch.tensor([2,3,-1,1])

a = torch.tensor(1.0,requires_grad=True)
b = torch.tensor(2.0,requires_grad=True)

learningRate = 0.01
loss_list = []
iterations = 500
previous_digit = 0
patience = 5
counter = 0
precision = 5

for i in range(iterations):
    Y_pred = forward(X)
    loss = calculationMSE(Y_pred, Y)
    loss_list.append(loss.item())
    loss.backward()

    # Update Parameters
    a.data = a.data - learningRate * a.grad.data
    b.data = b.data - learningRate * b.grad.data
    digit = getDecimalDigit(a.item(), precision)

    # Reset Gradient to 0 for next interation
    a.grad.data.zero_()
    b.grad.data.zero_()
    print(f'Iteration: {i} \n Loss: {loss.item():.4f} \n a: {a.item():.6f}\n b: {b.item():.6f}')
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
with torch.no_grad():
    plt.scatter(X.numpy(), Y.numpy(), color='blue', label='Target Points')
    plt.plot(X.numpy(), forward(X).numpy(), color='red', label='Learned Line')
plt.grid(True, color='y')
plt.legend()
plt.title(f'Final: y = {a.item():.2f}x + {b.item():.2f}')

plt.tight_layout()
plt.show()
