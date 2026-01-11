import torch
import torchvision.models as models

# Modell erstellen (z.B. AlexNet)
model = models.alexnet(pretrained=False)
model = model.cuda()

print(f"DEBUG: Model dtype is {next(model.parameters()).dtype}")
print(f"DEBUG: Allocated Memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")