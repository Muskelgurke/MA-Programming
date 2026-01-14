import torch
import torch.nn as nn
import torchvision.models as models
import pandas as pd
import helpers.model as model_helper
from helpers.datasets import get_sample_batch
from helpers.config_class import Config


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
            "d": d_h,
            "G": groups,
            "numel_params": sum(p.numel() for p in module.parameters()),  # Gewichte + Bias
            "numel_params_bytes": int,
            "bp_numel_activations": output.numel(), # Output Aktivierungen
            "bp_numel_activations_bytes": int,
            "bp_calc_activation": int,
            "bp_calc_activation_bytes": int,
            "fgd_calc_activation": int,
            "fgd_calc_activation_bytes": int
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

def calc_activation(row):
    # Fall 1: Linear Layer (Wir nutzen N_in wie in deiner Formel)
    BATCH_SIZE = row["BatchSize"]
    if row["Layer"] == "Linear":
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

    elif row["Layer"] == "BatchNorm2d":
        # BatchNorm behält die Input-Dimensionen bei
        feature_maps = BATCH_SIZE * int(row["N_out"]) * int(row["H_out"]) * int(row["W_out"])
        additional_params = 2 * int(row["N_out"])  # gamma und beta
        return feature_maps

    elif row["Layer"] in ["ReLU", "Sigmoid", "Tanh"]:
        # Aktivierungen: B * N_out * H_out * W_out
        # Da ich immer die Output dimensionen nehme zum Berechnen und RelU InputDimension = OutputDimension ist
        # kann ich hier einfach die Input Dimensionen nehmen
        if row["H_out"] == 0 or row["W_out"] == 0:
            params = int(row["N_out"])
        else:
            params = int(row["N_out"]) * int(row["H_out"]) * int(row["W_out"])
        return BATCH_SIZE * params
    return 0


pd.set_option('display.max_columns', None)  # Alle Spalten anzeigen
pd.set_option('display.width', 1000)    # Breite für Anzeige erhöhen
pd.set_option('display.max_colwidth', None)  # Keine Spaltenbreitenbegrenzung


print("\n------------- Analyse -------------")
dtype = 4 #float34
config = Config(
    cuda_device=0,
    random_seed=42,
    dataset_name="flower",
    learning_rate=0.001,
    epoch_total=10,
    batch_size=64,
    dataset_path="_Dataset",
    model_type="alexnet",
    training_method="bp",
    optimizer="Adam",
    loss_function="CrossEntropy",
    momentum=0.9,
    early_stopping_delta=0.001,
    memory_snapshot_epochs=[],
)

# Sample-Batch holen (läuft komplett auf CPU)
sample_x, sample_y = get_sample_batch(config)

print(f"Batch Shape: {sample_x.shape}")  # z.B. (64, 1, 28, 28)
print(f"Labels Shape: {sample_y.shape}")  # z.B. (64,)

sample_batch = sample_x.shape

model = model_helper.get_model(config=config, sample_batch= (sample_x,sample_y))

analyzer = ModelMemoryAnalyzer(model, input_size=sample_batch)
df_model = analyzer.analyze()
df_model = df_model.fillna(0)

# ------------------------------------------------
# Berechnung der Aktivierungen nach beiden Methoden BP FGD
df_model["bp_calc_activation"] = df_model.apply(calc_activation, axis=1)
# --------------------------------------------------
# Forward Gradient
df_model["fgd_calc_activation"] = 0
max_idx = df_model["bp_calc_activation"].idxmax()
df_model.loc[max_idx, "fgd_calc_activation"] = df_model["bp_calc_activation"].max() * 2
df_model["fgd_calc_activation_bytes"] = df_model["fgd_calc_activation"] * dtype



df_model["bp_calc_activation_bytes"] = df_model["bp_calc_activation"] * dtype
df_model["numel_params_bytes"] = df_model["numel_params"] * dtype
df_model["bp_numel_activations_bytes"] = df_model["bp_numel_activations"] * dtype


# --- 1. Die Summen berechnen ---
sum_params = df_model["numel_params"].sum()
sum_params_bytes = df_model["numel_params_bytes"].sum()
sum_act_hook = df_model["bp_numel_activations"].sum()
sum_act_hook_bytes = df_model["bp_numel_activations_bytes"].sum()
sum_act_calc = df_model["bp_calc_activation"].sum()
sum_act_calc_bytes = df_model["bp_calc_activation_bytes"].sum()
sum_act_fgd_activation = df_model["fgd_calc_activation"].sum()  # Summe = nur der eine Wert
sum_act_fgd_bytes = sum_act_fgd_activation * dtype
sum_act_fgd_forward = sum_act_fgd_activation + sum_params
sum_act_fgd_forward_bytes = sum_act_fgd_forward * dtype
# --- 2. Eine neue Zeile als DataFrame erstellen ---
# Wir setzen nur "Layer" auf "TOTAL" und die Summen-Spalten.
# Alle anderen Spalten (Kernel, Stride etc.) bleiben automatisch NaN (leer).
total_row = pd.DataFrame([{
    "Layer": "TOTAL",
    "numel_params": sum_params,
    "numel_params_bytes":sum_params_bytes,
    "bp_numel_activations": sum_act_hook,
    "bp_numel_activations_bytes":sum_act_hook_bytes,
    "bp_calc_activation": sum_act_calc,
    "bp_calc_activation_bytes": sum_act_calc_bytes,
    "fgd_calc_activation": sum_act_fgd_activation,
    "fgd_calc_activation_bytes": sum_act_fgd_bytes,
    "fgd_calc_forward": sum_act_fgd_forward,
    "fgd_calc_forward_bytes": sum_act_fgd_forward_bytes
}])

# --- 3. Zusammenfügen ---
# ignore_index=True sorgt dafür, dass der Index fortlaufend nummeriert wird (0, 1, ..., 11, 12)
df_final = pd.concat([df_model, total_row], ignore_index=True)
df_display = df_final.replace(0.0, "-")
df_display = df_display.fillna("-")


df_final.to_csv(f"/home/muskelgurke/PycharmProjects/JupyterProject/data/training_sessions/results/{config.model_type}/_theoretisch/model_param_analysis_{config.dataset_name}.csv", index=False)
print("-" * 50)
print("Tabelle mit Summenzeile:")
print("-" * 50)

print(df_display)

