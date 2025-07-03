import torch
import matplotlib.pyplot as plt
import yaml


with open("../config.yaml", "r") as file:
    config = yaml.safe_load(file)

learningRate = config["learningRate"]
iterations = config["iterations"]
patience = config["patience"]
precision = config["precision"]
value_a = config["a"]
value_b = config["b"]
data_x = config["x"]
data_y = config["y"]

X = torch.tensor(data_x)
Y = torch.tensor(data_y)

def linearFkt(a,b,x):
    return a * x + b

def calculationMSE(a, b):
    Y_pred = linearFkt(a, b, X)
    return torch.mean((Y_pred - Y) ** 2)

def startTrain(plot = False):
    a = torch.tensor(value_a, requires_grad=True, dtype=torch.float32)
    b = torch.tensor(value_b, requires_grad=True, dtype=torch.float32)
    print(f'\nStarting training lReg with torch...\n'
          f'\tYour Funktion: y = {a.item():.6f} * X + {b.item():.6f}\n'
          f'\tIterations: {iterations} \n '
          f'\tlearning rate: {learningRate} \n '
          f'\tprecision: {precision}')

    loss_list = []
    previous_digit = 0
    counter = 0

    for i in range(iterations):
        loss = calculationMSE(a,b)
        loss_list.append(loss.item())
        loss.backward()
        # print(f'Iteration: {i + 1} \n Loss: {loss.item():.4f} \n a: {a.item():.6f}\n b: {b.item():.6f}')
        # Update Parameters
        a.data = a.data - learningRate * a.grad.data
        b.data = b.data - learningRate * b.grad.data
        digit = getDecimalDigit(a.item(), precision)
        #print(f'Iteration: {i} \n Loss: {loss.item():.4f} \n a: {a.item():.6f}\n b: {b.item():.6f}')
        a.grad.data.zero_()
        b.grad.data.zero_()
        # print(f'Your New Funktion: y = {a.item():.6f} * X + {b.item():.6f}\n')
        if digit == previous_digit:
            counter += 1
        else:
            counter = 0
        previous_digit = digit
        if counter >= patience:
            print(f"\tStopping early at epoch {i} as the {precision}-th decimal hasn´t changed")
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
        with torch.no_grad():
            plt.scatter(X.numpy(), Y.numpy(), color='blue', label='Target Points')
            plt.plot(X.numpy(), linearFkt(a,b,X).numpy(), color='red', label='Learned Line')
        plt.grid(True, color='y')
        plt.legend()
        plt.title(f'Final: y = {a.item():.2f}x + {b.item():.2f}')

        plt.tight_layout()
        plt.show()

    return a.item(), b.item()

def getDecimalDigit(value,position):
    as_str = f'{value:.{position + 2}f}'
    return int(as_str.split(".")[1][position - 1])

