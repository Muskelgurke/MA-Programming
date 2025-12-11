import torch
import torch.nn as nn
from torchvision.models import resnet18


# --- 1. Dein Analytisches Modell (Erweitert für Conv2d) ---
class AnalyticalMemoryBP:
    def __init__(self, model: nn.Module):
        self.model = model
        self.bp_memory_bytes = 0
        self.hooks = []

    def _register_hooks(self):
        def hook_fn(module, input, output):
            # Input[0] ist der Tensor, der in den Layer reinging
            x = input[0]

            # --- REGELWERK ---
            # Linear & Conv2d speichern den Input für Backward
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                self.bp_memory_bytes += x.numel() * x.element_size()

            # BatchNorm speichert Input (und oft Output für Statistiken, hier vereinfacht Input)
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                self.bp_memory_bytes += x.numel() * x.element_size()

            # HIER WICHTIG: ReLU (inplace=False) speichert oft eine Bitmaske oder den Output.
            # PyTorch ist hier effizient. Wir nehmen konservativ an, es speichert den Output.
            # Wenn du es genauer willst, musst du tiefer graben.
            # elif isinstance(module, nn.ReLU) and not module.inplace:
            #     self.bp_memory_bytes += output.numel() * output.element_size()

        for module in self.model.modules():
            # Nur Leaf-Module hooken
            if len(list(module.children())) == 0:
                self.hooks.append(module.register_forward_hook(hook_fn))

    def predict_memory(self, input_tensor):
        self.bp_memory_bytes = 0
        self.hooks = []
        self._register_hooks()

        # Dummy Pass (nur Shapes sammeln)
        with torch.no_grad():
            self.model(input_tensor)

        for h in self.hooks:
            h.remove()

        return self.bp_memory_bytes


# --- 2. Die Empirische Messung (Realität) ---
def measure_real_memory(model, input_tensor):
    if not torch.cuda.is_available():
        raise RuntimeError("Für präzise Messung wird eine GPU benötigt!")

    device = torch.device("cuda")
    model.to(device)
    input_tensor = input_tensor.to(device)
    # Warm-up Lauf damit CUDA initialisiert ist
    _ = model(input_tensor)
    # Garbage Collection & Cache leeren für sauberen Start
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # 1. Baseline messen (Modell-Gewichte sind schon drauf)
    mem_before = torch.cuda.memory_allocated(device)

    # 2. Forward Pass
    output = model(input_tensor)

    # 3. Speicher messen (Modell + Activations)
    mem_after = torch.cuda.memory_allocated(device)

    # Die Differenz sind die Activations
    real_activation_memory = mem_after - mem_before

    return real_activation_memory


# --- 3. Der Vergleich ---
def run_comparison():
    print(f"{'Modell / Layer':<20} | {'Theorie (Bytes)':<15} | {'Realität (Bytes)':<15} | {'Diff (%)':<10}")
    print("-" * 70)

    test_cases = [
        ("Linear (10->10)", nn.Linear(10, 10, bias=False), (1, 10)),
        ("Linear Large", nn.Linear(1000, 500, bias=False), (64, 1000)),  # Batch 64
        ("Conv2d Simple", nn.Conv2d(3, 16, kernel_size=3, padding=1), (1, 3, 32, 32)),
        # ResNet18 (komplexer Test)
        ("ResNet18 (Teil)", nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)  # Inplace spart Speicher!
        ), (1, 3, 224, 224))
    ]

    for name, model, input_shape in test_cases:
        # Input erstellen
        input_tensor = torch.randn(*input_shape)

        # 1. Theorie
        analytical_tool = AnalyticalMemoryBP(model)
        theo_bytes = analytical_tool.predict_memory(input_tensor)

        # 2. Realität (nur wenn GPU da ist, sonst 0 setzen)
        try:
            real_bytes = measure_real_memory(model, input_tensor)
        except RuntimeError:
            print("Keine GPU gefunden -> Überspringe Real-Messung")
            real_bytes = 0  # Fake für Demo

        # 3. Auswertung
        diff_percent = 0
        if real_bytes > 0:
            diff_percent = abs(theo_bytes - real_bytes) / real_bytes * 100

        print(f"{name:<20} | {theo_bytes:<15} | {real_bytes:<15} | {diff_percent:.2f}%")


if __name__ == "__main__":
    run_comparison()