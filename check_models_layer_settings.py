import torch
import torch.nn as nn
import torchvision.models as models
import pandas as pd

# Deine LeNet5 Klasse (aus deiner Frage kopiert, damit das Skript vollständig ist)
class LeNet5(nn.Module):
    def __init__(self, num_input_channel, num_classes, input_size):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(num_input_channel, 6, 5)
        self.activation1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.activation2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        feature_map_size = ((input_size - 4) // 2 - 4) // 2
        flattend_size = 16 * feature_map_size * feature_map_size
        self.fc1 = nn.Linear(flattend_size, 120)
        self.activation3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.activation4 = nn.ReLU()
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        y = self.conv1(x)
        y = self.activation1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.activation2(y)
        y = self.pool2(y)
        y = torch.flatten(y, 1)
        y = self.fc1(y)
        y = self.activation3(y)
        y = self.fc2(y)
        y = self.activation4(y)
        y = self.fc3(y)
        return y


# --- KERNELEMENT: Der Analyser ---
class ModelMemoryAnalyzer:
    """
    Nimmt ein PyTorch Modell und eine Input-Größe.
    Führt einen Dummy-Forward-Pass durch und sammelt Layer-Informationen
    wie Input/Output-Dimensionen, Kernelgrößen, Strides, Paddings, Dilation Groups
    und Parameteranzahl.
    Gibt die gesammelten Daten als Pandas DataFrame zurück.
    Nur für conv2d, linear, pooling und Aktivierungsschichten.
    """
    def __init__(self, model, input_size=(1, 3, 224, 224)):
        self.model = model
        self.input_size = input_size
        self.layer_data = []
        self.hooks = []

    def _to_pair(self, val):
        """Wandelt int in (val, val) um, falls nötig"""
        if isinstance(val, int):
            return (val, val)
        return val

    def _hook_fn(self, module, input, output):
        # Input/Output Dimensionen ermitteln
        x_in = input[0]

        layer_type = module.__class__.__name__
        h_in, w_in = None, None
        h_out, w_out = None, None
        n_in, n_out = None, None



        if len(x_in.shape) == 4: #Conv, Pool, Activation
            b, n_in, h_in, w_in = x_in.shape
        elif len(x_in.shape) == 2:  # Linear Layer Input (Flattened)
            b, n_in = x_in.shape
        else:
            return

        if len(output.shape) == 4:
            _, n_out, h_out, w_out = output.shape
        elif len(output.shape) == 2:
            _, n_out = output.shape
        else:
            return

        # Standardwerte
        k_h, k_w = None, None
        s_h, s_w = None, None
        p_h, p_w = None, None
        d_h, d_w = None, None
        groups = 1
        layer_type = module.__class__.__name__


        # Spezifische Attribute extrahieren (Genau das, was im print() steht)
        if isinstance(module, (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d)):
            k = self._to_pair(module.kernel_size)
            s = self._to_pair(module.stride)
            p = self._to_pair(module.padding)

            k_h, k_w = k
            s_h, s_w = s
            p_h, p_w = p

            # Dilation (nur bei Conv üblich, manchmal bei Pool)
            if hasattr(module, 'dilation'):
                d = self._to_pair(module.dilation)
                d_h, d_w = d
            else:
                d_h, d_w = 1, 1

            if isinstance(module, nn.Conv2d):
                groups = module.groups

        # Datenstruktur für deine Formeln
        self.layer_data.append({
            "Layer": layer_type,
            "BatchSize": b,
            "N_in": n_in, "N_out": n_out,
            "H_in": h_in, "W_in": w_in,
            "H_out": h_out, "W_out": w_out,
            "K_h": k_h, "K_w": k_w,  # Kernel Size
            "S_h": s_h, "S_w": s_w,  # Stride
            "P_h": p_h, "P_w": p_w,  # Padding
            "d": d_h,  # Dilation (Vereinfacht d_h)
            "G": groups,  # Groups
            "NUMEL_Params_bytes": sum(p.numel() for p in module.parameters()),  # Gewichte + Bias
            "NUMEL_Activations_bytes": output.numel()  # Output Aktivierungen
        })


    def analyze(self):
        # 1. Hooks an alle relevanten Layer hängen
        # Wir gehen rekursiv durch alle Module, nehmen aber nur die "Blätter" (Leaf Modules)
        for name, module in self.model.named_modules():
            # Filter: Wir wollen nur echte Rechenschichten, keine Container wie Sequential
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d,
                                   nn.MaxPool2d, nn.AvgPool2d, nn.ReLU,
                                   nn.Sigmoid, nn.Tanh)):
                self.hooks.append(module.register_forward_hook(self._hook_fn))

        # 2. Dummy Pass durchführen (löst die Hooks aus)
        dummy_input = torch.zeros(self.input_size)
        try:
            with torch.no_grad():
                self.model(dummy_input)
        except Exception as e:
            print(f"Fehler beim Forward Pass: {e}")
        finally:
            # 3. Aufräumen: Hooks entfernen
            for h in self.hooks:
                h.remove()

        return pd.DataFrame(self.layer_data)

def calc_parameter_memory(row):
    # Fall 1: Linear Layer
    if row["Layer"] == "Linear":
        # N_in * N_out * 4 Bytes + N_out * 4 Bytes (Bias)
        params = int(row["N_in"]) * int(row["N_out"]) + int(row["N_out"])
        return params

    # Fall 2: Conv Layer
    elif row["Layer"] == "Conv2d":


        params = (int(row["K_h"]) * int(row["K_w"]) *
         (int(row["N_in"]) // row["G"]) * int(row["N_out"]) +
         int(row["N_out"]))
        return params
    return 0


def calc_activation_memory(row):
    # Fall 1: Linear Layer (Wir nutzen N_in wie in deiner Formel)
    BATCH_SIZE = row["BatchSize"]
    if row["Layer"] == "Linear":
        # B * N_in * 4 Bytes
        return BATCH_SIZE * int(row["N_out"])

    # Fall 2: Conv/Pool Layer (Wir nutzen H_out * W_out * C_out wie in deiner Formel)
    elif row["Layer"] == "Conv2d":
        h_out = (row["H_in"] + 2 * row["P_h"] - row["d"] * (int(row["K_h"]) - 1) - 1) // row["S_h"] + 1
        w_out = (row["W_in"] + 2 * row["P_w"] - row["d"] * (int(row["K_w"]) - 1) - 1) // row["S_w"] + 1
        # n_out ist die Anzahl der Filterkanäle, dieser ist ein wert der
        # Designspeizifisch ist und nicht von der Eingabe abhängt.


        return BATCH_SIZE * int(row["N_out"]) * int(h_out) * int(w_out)

    elif row["Layer"] in ["MaxPool2d", "AvgPool2d"]:
        h_out = (row["H_in"] + 2 * row["P_h"] - row["d"] * (int(row["K_h"]) - 1) - 1) // row["S_h"] + 1
        w_out = (row["W_in"] + 2 * row["P_w"] - row["d"] * (int(row["K_w"]) - 1) - 1) // row["S_w"] + 1
        return BATCH_SIZE * int(row["N_out"]) * int(h_out) * int(w_out)

    elif row["Layer"] in ["ReLU", "Sigmoid", "Tanh"]:
        # Aktivierungen: B * N_out * H_out * W_out
        # Da ich immer die Output dimensionen nehme zum Berechnen und RelU InputDimension = OutputDimension ist
        # kann ich hier einfach die Input Dimensionen nehmen
        if row["H_out"] >0 or row["W_out"] > 0:
            params = int(row["N_out"])
        else:
            params = int(row["N_out"]) * int(row["H_out"]) * int(row["W_out"])
        return BATCH_SIZE * params
    return 0

# Test für LeeNet5
pd.set_option('display.max_columns', None)  # Alle Spalten anzeigen
pd.set_option('display.width', 1000)    # Breite für Anzeige erhöhen
pd.set_option('display.max_colwidth', None)  # Keine Spaltenbreitenbegrenzung
"""
# 1. AlexNet (Standard PyTorch)
print("--- Analyse: AlexNet ---")
alexnet = models.alexnet(pretrained=False)
analyzer_alex = ModelMemoryAnalyzer(alexnet, input_size=(1, 3, 224, 224))
df_alex = analyzer_alex.analyze()

# Filtern wir mal genau die Spalten, die du in deiner Arbeit erklärt hast:
print(df_alex[cols_interest].head(10))  # Zeige die ersten Schichten (Features)
"""


# 2. LeNet5 (Dein Custom Model)
print("\n--- Analyse: LeNet5 ---")
# Hinweis: LeNet erwartet 32x32 Input laut deiner Klasse
lenet = LeNet5(num_input_channel=1, num_classes=10, input_size=32)
analyzer_lenet = ModelMemoryAnalyzer(lenet, input_size=(1, 1, 32, 32))
df_lenet = analyzer_lenet.analyze()
df_lenet = df_lenet.fillna(0)

df_lenet["Calc_Activation_bytes"] = df_lenet.apply(calc_activation_memory, axis=1)
df_display = df_lenet.copy()
df_display = df_display.replace(0.0, "-")

print(df_display)

