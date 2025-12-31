import torch
import torchvision.models as models

# ResNet18 Modell erstellen
#model = models.resnet18(pretrained=False)
model = models.vgg16(pretrained=False)

# Modell-Struktur ausgeben
#print("ResNet18 Modell-Struktur:")
print("=" * 80)
print(model)
print("\n" + "=" * 80)

# Detaillierte Layer-Informationen
print("\nLayer-Details:")
print("=" * 80)
for name, layer in model.named_modules():
    if name:  # Skip das Hauptmodell selbst
        print(f"{name}: {layer.__class__.__name__}")

# Anzahl der Parameter
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("\n" + "=" * 80)
print(f"Gesamtanzahl Parameter: {total_params:,}")
print(f"Trainierbare Parameter: {trainable_params:,}")