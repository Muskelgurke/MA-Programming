import torch
torch.manual_seed(42)
x= [1.0, 2.0, 3.0]
y= [0.0, 1.0]



model_bp = torch.nn.Linear(3, 2) # 3 input features, 2 output features
with torch.no_grad():
    model_bp.weight.data = torch.tensor([[1.0, 2.0, 1.0],  # Gewichte für Output 1
                                         [2.0, 1.0, 2.0]])  # Gewichte für Output 2
    model_bp.bias.data = torch.tensor([1.0, 2.0])           # Bias-Werte

#print(f'model Parameter: \t \n{dict(model_bp.named_parameters())}\n')

loss_function_bp = torch.nn.MSELoss()
optimizer_bp = torch.optim.SGD(model_bp.parameters(), lr=0.01)

predictions = model_bp(torch.tensor([x]))
#print(f'{predictions = }\n')
loss_bp = loss_function_bp(predictions, torch.tensor([y]))
print(f'{loss_bp}\n')
loss_bp.backward()
print(f'Gradients after backward pass: \n{model_bp.weight.grad = }\n {model_bp.bias.grad = }\n')
optimizer_bp.step()
print(f'Backpropagation 1-Opti. Step: \n {dict(model_bp.named_parameters())}')


# ---
print('--- Forward Gradient Descent Meine Methode mit functional_call ---\n')

model_fgd = torch.nn.Linear(3, 2) # 3 input features, 2 output features
with torch.no_grad():
    model_fgd.weight.data = torch.tensor([[1.0, 2.0, 1.0],  # Gewichte für Output 1
                                         [2.0, 1.0, 2.0]])  # Gewichte für Output 2
    model_fgd.bias.data = torch.tensor([1.0, 2.0])           # Bias-Werte

optimizer_fgd = torch.optim.SGD(model_fgd.parameters(), lr=0.001)
named_params = dict(model_fgd.named_parameters())
params = tuple(named_params.values())
names = tuple(named_params.keys())

#v_params = tuple([torch.randn_like(p) for p in params])
v_params = tuple([torch.tensor([[0.4617, 0.2674, 0.5349], # Gewichte für Output 1)
                                         [0.8094, 1.1103, -1.6898]]),  # Gewichte für Output 2
                      torch.tensor([-0.9890, 0.9580])])           # Bias-Werte
print(f'v_params = {v_params}\n')


def loss_fn(params_tuple, inputs, targets):
    # Reconstruct parameter dict from tuple
    params_dict = dict(zip(names, params_tuple))
    output = torch.func.functional_call(model_fgd, params_dict, inputs)
    print(f'output = {output}\n')
    return torch.nn.functional.mse_loss(output, targets)

loss, dir_der = torch.func.jvp(
                    lambda params: loss_fn(params, torch.tensor([x]), torch.tensor([y])),
                    (params,),
                    (v_params,)
                )
print(f'Loss: {loss}\n Directional Derivative: {dir_der}\n')

with torch.no_grad():
    for j, param in enumerate(model_fgd.parameters()):
        print(f'{model_fgd.named_parameters()}')
        print(f'{v_params[j]=}, {dir_der=}')
        param.grad = dir_der * v_params[j]
        print(param.grad)

optimizer_fgd.step()

print(f'FGD 1-Opti. Step: \n {dict(model_fgd.named_parameters())}')
print(f'Backpropagation 1-Opti. Step: \n {dict(model_bp.named_parameters())}')