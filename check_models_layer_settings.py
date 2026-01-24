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

        if isinstance(module, (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d)):
            k = self._to_pair(module.kernel_size)
            s = self._to_pair(module.stride)
            p = self._to_pair(module.padding)

            k_h, k_w = k
            s_h, s_w = s
            p_h, p_w = p

            if hasattr(module, 'dilation'):
                d = self._to_pair(module.dilation)
                d_h, d_w = d
            else:
                d_h, d_w = 1, 1

            if isinstance(module, nn.Conv2d):
                groups = module.groups

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
            "model_params": sum(p.numel() for p in module.parameters()),  # Gewichte + Bias
            "model_params_bytes": int,
            "bp_calc_activation": int,
            "bp_calc_activation_bytes": int,
            "fgd_calc_activation": int,
            "fgd_calc_activation_bytes": int
        })


    def analyze(self):
        # 1. Hooks an alle relevanten Layer hängen
        # rekursiv durch alle Module
        for name, module in self.model.named_modules():
            # Filter: Nur relevante Layer
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d,
                                   nn.MaxPool2d, nn.AvgPool2d, nn.ReLU,
                                   nn.Sigmoid, nn.Tanh)):
                self.hooks.append(module.register_forward_hook(self._hook_fn))

        dummy_input = torch.zeros(self.input_size)
        try:
            with torch.no_grad():
                self.model(dummy_input)
        except Exception as e:
            print(f"Fehler beim Forward Pass: {e}")
        finally:
            for h in self.hooks:
                h.remove()

        return pd.DataFrame(self.layer_data)

def calc_activation(row):
    BATCH_SIZE = row["BatchSize"]
    if row["Layer"] == "Linear":
        return BATCH_SIZE * int(row["N_out"])

    elif row["Layer"] == "Conv2d":
        h_out = (row["H_in"] + 2 * row["P_h"] - row["d"] * (int(row["K_h"]) - 1) - 1) // row["S_h"] + 1
        w_out = (row["W_in"] + 2 * row["P_w"] - row["d"] * (int(row["K_w"]) - 1) - 1) // row["S_w"] + 1


        return BATCH_SIZE * int(row["N_out"]) * int(h_out) * int(w_out)

    elif row["Layer"] in ["MaxPool2d", "AvgPool2d"]:
        h_out = (row["H_in"] + 2 * row["P_h"] - row["d"] * (int(row["K_h"]) - 1) - 1) // row["S_h"] + 1
        w_out = (row["W_in"] + 2 * row["P_w"] - row["d"] * (int(row["K_w"]) - 1) - 1) // row["S_w"] + 1
        return BATCH_SIZE * int(row["N_out"]) * int(h_out) * int(w_out)


    elif row["Layer"] == "BatchNorm2d":

        feature_maps = BATCH_SIZE * int(row["N_out"]) * int(row["H_out"]) * int(row["W_out"])
        additional_params = 2 * int(row["N_out"])  # gamma und beta
        return feature_maps

    elif row["Layer"] in ["ReLU", "Sigmoid", "Tanh"]:
        # Aktivierungen: B * N_out * H_out * W_out
        if row["H_out"] == 0 or row["W_out"] == 0:
            params = int(row["N_out"])
        else:
            params = int(row["N_out"]) * int(row["H_out"]) * int(row["W_out"])
        return BATCH_SIZE * params
    return 0


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)

import pandas as pd

print("\n------------- Analyse -------------")
dtype = 4  # float32 (4 Bytes)
models = ['alexnet', 'vgg16', 'resnet18', 'resnet34', 'mobilenet', 'densenet', 'efficientnet', 'lenet']
datasets = ['mnist', 'fashionmnist', 'cifar10', 'cifar100', 'flower', 'pet']
optimizers = ['adam', 'sgd']

# Für jede Kombination durchführen
for model_type in models:
    for dataset_name in datasets:
        for optimizer_name in optimizers:
            print(f"\n{'=' * 60}")
            print(f"Verarbeite: {model_type} | {dataset_name} | {optimizer_name}")
            print(f"{'=' * 60}")

            try:
                config = Config(
                    cuda_device=0,
                    random_seed=42,
                    dataset_name=dataset_name, # z.B. "mnist", "cifar10"
                    learning_rate=0.001,
                    epoch_total=10,
                    batch_size=64,
                    dataset_path="_Dataset",
                    model_type=model_type,
                    training_method="bp",
                    optimizer=optimizer_name,  # z.B. "Adam" oder "Sgd"
                    loss_function="CrossEntropy",
                    momentum=0.9,
                    early_stopping_delta=0.001,
                    memory_snapshot_epochs=[],
                )

                # Sample-Batch holen
                sample_x, sample_y = get_sample_batch(config)
                sample_batch = sample_x.shape

                # Modell laden und analysieren
                model = model_helper.get_model(config=config, sample_batch=(sample_x, sample_y))
                analyzer = ModelMemoryAnalyzer(model, input_size=sample_batch)
                df_model = analyzer.analyze()
                df_model = df_model.fillna(0)

                # ---------------------------------------------------------
                # BERECHNUNGEN
                # ---------------------------------------------------------

                # 1. Aktivierungen
                df_model["bp_calc_activation"] = df_model.apply(calc_activation, axis=1)

                # FGD Aktivierung (Heuristik: Max Layer * 2 für Buffer)
                df_model["fgd_calc_activation"] = 0
                max_idx = df_model["bp_calc_activation"].idxmax()
                df_model.loc[max_idx, "fgd_calc_activation"] = df_model["bp_calc_activation"].max() * 2

                # Byte-Konvertierung
                df_model["fgd_calc_activation_bytes"] = df_model["fgd_calc_activation"] * dtype
                df_model["bp_calc_activation_bytes"] = df_model["bp_calc_activation"] * dtype
                df_model["model_params_bytes"] = df_model["model_params"] * dtype

                # Summen berechnen
                m_model = df_model["model_params"].sum()
                m_model_bytes = df_model["model_params_bytes"].sum()

                acti_bp = df_model["bp_calc_activation"].sum()
                m_acti_bp_bytes = df_model["bp_calc_activation_bytes"].sum()

                acti_fgd = df_model["fgd_calc_activation"].sum()
                m_acti_fgd_bytes = acti_fgd * dtype

                # ---------------------------------------------------------
                # ANALYTISCHES MODELL (OPTIMIZER ABHÄNGIG)
                # ---------------------------------------------------------

                m_grads = m_model

                # Optimizer States (Momentum Buffer etc.)
                if optimizer_name.lower() == 'adam':
                    m_opt_state = 2 * m_model
                else: # SGD oder andere
                    m_opt_state = 2 * m_model


                # 1.Theorie: 1x activierungen + 1x Gradienten
                m_dynamic_bp = acti_bp + m_model

                # acti + pertubation
                # (FALSCH) 1.THEORIE: 1x m_model 2x acti_fgd ,weil Forward layer braucht immer die aktivierungen des vorherigen layers. Dannkann erst weggeworfen werden.
                # 2.THEORIE: 1x acti_fgd 2x m_model, activierungen 1x Pertubation 1x BufferStates
                m_dynamic_fgd = acti_fgd + m_model + m_model

                # ---------------------------------------------------------

                # Bytes berechnen
                m_dynamic_bp_bytes = m_dynamic_bp * dtype
                m_dynamic_fgd_bytes = m_dynamic_fgd * dtype

                # Total Row erstellen
                total_row = pd.DataFrame([{
                    "Layer": "TOTAL",
                    "m_model": m_model,
                    "m_model_bytes": m_model_bytes,
                    "acti_bp": acti_bp,
                    "m_acti_bp_bytes": m_acti_bp_bytes,
                    #"m_forward_bp_bytes": m_acti_bp_bytes,  # Legacy Name checken?
                    "m_dynamic_bp": m_dynamic_bp,
                    "m_dynamic_bp_bytes": m_dynamic_bp_bytes,
                    "acti_fgd": acti_fgd,
                    "m_acti_fgd_bytes": m_acti_fgd_bytes,
                    #"m_forward_fgd_bytes": m_dynamic_fgd_bytes,  # Legacy Name checken?
                    "m_dynamic": m_dynamic_fgd,
                    "m_dynamic_bytes": m_dynamic_fgd_bytes
                }])

                df_final = pd.concat([df_model, total_row], ignore_index=True)

                # Bereinigen
                df_save = (df_final.drop(
                    columns=["bp_calc_activation",
                             "model_params",
                             "fgd_calc_activation",
                             "m_total_fgd",
                             "K_h", "K_w",
                             "S_h", "S_w",
                             "P_h", "P_w",
                             "d", "G"
                             ], errors='ignore')  # errors='ignore' falls Spalte nicht existiert
                           .replace(0.0, "-").fillna("-"))

                # Datei speichern - Jetzt mit Optimizer im Namen
                output_filename = f"model_param_analysis_{dataset_name}_{optimizer_name}.csv"
                output_dir = f"/home/muskelgurke/PycharmProjects/JupyterProject/data/training_sessions/results/{model_type}/_theoretisch/"

                # Sicherstellen, dass Verzeichnis existiert (optional, aber gut)
                import os

                os.makedirs(output_dir, exist_ok=True)

                output_path = os.path.join(output_dir, output_filename)

                df_save.to_csv(output_path, index=False)
                print(f"✓ Gespeichert: {output_filename}")

            except Exception as e:
                print(f"✗ Fehler bei {model_type}/{dataset_name}/{optimizer_name}: {e}")
                continue

print(f"\n{'=' * 60}")
print("Alle Kombinationen verarbeitet!")
print(f"{'=' * 60}")
