import torch
import torch.autograd.forward_ad as fwAD

# Reproduzierbarkeit
torch.manual_seed(42)
x = [1.0, 2.0, 3.0]
y = [0.0, 1.0]

# ==========================================
# 1. Standard Backpropagation (Ground Truth)
# ==========================================
print('--- 1. Standard Backpropagation ---\n')

model_bp = torch.nn.Linear(3, 2)
with torch.no_grad():
    model_bp.weight.copy_(torch.tensor([[1.0, 2.0, 1.0], [2.0, 1.0, 2.0]]))
    model_bp.bias.copy_(torch.tensor([1.0, 2.0]))

loss_function_bp = torch.nn.MSELoss()
# Wir nutzen hier SGD mit lr=0.01 zum Vergleich
optimizer_bp = torch.optim.SGD(model_bp.parameters(), lr=0.01)

# Forward & Backward
predictions = model_bp(torch.tensor([x]))
loss_bp = loss_function_bp(predictions, torch.tensor([y]))
loss_bp.backward()

print(f'Loss BP: {loss_bp.item()}')
print(f'Gradients BP (Weight):\n{model_bp.weight.grad}')
print(f'Gradients BP (Bias):\n{model_bp.bias.grad}')

optimizer_bp.step()
print(f'\nWeights after BP Step:\n{model_bp.weight}\n')


# ==========================================
# 2. Forward Gradient (Dual Mode / FROG Style)
# ==========================================
print('-' * 40)
print('--- 2. Forward Gradient Descent (Dual Mode) ---\n')

model_fgd = torch.nn.Linear(3, 2)
with torch.no_grad():
    model_fgd.weight.copy_(torch.tensor([[1.0, 2.0, 1.0], [2.0, 1.0, 2.0]]))
    model_fgd.bias.copy_(torch.tensor([1.0, 2.0]))

# WICHTIG: Gleiche LR wie im BP Beispiel oben nutzen, sonst sind Zahlen nicht vergleichbar
optimizer_fgd = torch.optim.SGD(model_fgd.parameters(), lr=0.01)

# Fest definierte Perturbations (v), damit wir vergleichen können
# (Dieselben Werte wie in deinem Beispiel)
v_weight = torch.tensor([[0.4617, 0.2674, 0.5349], [0.8094, 1.1103, -1.6898]])
v_bias = torch.tensor([-0.9890, 0.9580])

print(f'Perturbation v (Weight):\n{v_weight}')
print(f'Perturbation v (Bias):\n{v_bias}\n')

# --- SCHRITT A: Parameter "Sichern" ---
# Wir speichern Referenzen auf die echten Parameter, weil wir sie gleich aus dem Modell löschen
params_store = {
    'weight': model_fgd.weight,
    'bias': model_fgd.bias
}

# Wir löschen die Parameter aus dem Modell, um Platz für Dual Tensors zu machen
del model_fgd.weight
del model_fgd.bias

# --- SCHRITT B: Forward AD Pass ---
with fwAD.dual_level():
    # 1. Dual Tensors erstellen und injizieren
    # make_dual(primal, tangent) verbindet Gewicht mit Störung
    dual_weight = fwAD.make_dual(params_store['weight'], v_weight)
    dual_bias = fwAD.make_dual(params_store['bias'], v_bias)

    # Injektion: Wir setzen die Dual Tensors als normale Attribute
    model_fgd.weight = dual_weight
    model_fgd.bias = dual_bias

    # 2. Forward Pass
    # Das Modell rechnet jetzt automatisch mit Dual-Zahlen
    output_dual = model_fgd(torch.tensor([x]))
    print(f'Output (Primal): {fwAD.unpack_dual(output_dual).primal}')

    # Loss berechnen (auch Loss ist jetzt Dual)
    # Wichtig: MSELoss muss funktional aufgerufen werden oder instanziiert sein
    loss_dual = torch.nn.functional.mse_loss(output_dual, torch.tensor([y]))

    # 3. Unpack Results
    unpacked_loss = fwAD.unpack_dual(loss_dual)
    loss_val = unpacked_loss.primal.item()
    jvp = unpacked_loss.tangent.item()  # Das ist (nabla_L * v)

    print(f'Loss FGD: {loss_val}')
    print(f'Directional Derivative (JVP): {jvp}')

# --- SCHRITT C: Restore & Update ---

# WICHTIG: Original-Parameter wiederherstellen!
# Wir löschen die Dual Tensors...
del model_fgd.weight
del model_fgd.bias
# ...und setzen die originalen Parameter-Objekte zurück
model_fgd.weight = params_store['weight']
model_fgd.bias = params_store['bias']

# Gradienten schreiben: grad = (dL/dv) * v = jvp * v
with torch.no_grad():
    model_fgd.weight.grad = jvp * v_weight
    model_fgd.bias.grad = jvp * v_bias

print(f'\nEstimated Gradients FGD (Weight):\n{model_fgd.weight.grad}')
print(f'Estimated Gradients FGD (Bias):\n{model_fgd.bias.grad}')

optimizer_fgd.step()
print(f'\nWeights after FGD Step:\n{model_fgd.weight}')

# ==========================================
# Vergleich
# ==========================================
print('\n' + '-' * 40)
print('VERGLEICH DER GEWICHTE (Differenz):')
print(f'Weight Diff: \n{model_bp.weight - model_fgd.weight}')
print('(Hinweis: Da FGD eine Schätzung ist, werden die Werte NICHT identisch sein,')
print(' aber sie sollten sich in eine ähnliche Richtung bewegt haben, falls v zufällig gut war.)')