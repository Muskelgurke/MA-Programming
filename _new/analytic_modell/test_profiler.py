import torch
import os
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity


def profile_training():
    """
    Profiles real training with multiple epochs (not just inference).
    """
    # 1. Set up a simple model, loss function, and optimizer
    model = models.resnet18(num_classes=10).cuda() if torch.cuda.is_available() else models.resnet18(num_classes=10)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    model.train()  # Set model to training mode

    # 2. Create dummy training data (simulate a small dataset)
    batch_size = 32
    num_batches = 10  # Simulate 10 batches

    # Generate synthetic dataset
    train_data = []
    for _ in range(num_batches):
        inputs = torch.randn(batch_size, 3, 224, 224)
        labels = torch.randint(0, 10, (batch_size,))
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
        train_data.append((inputs, labels))

    # 3. Configure and run the profiler
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    num_epochs = 3  # Train for 3 epochs

    # Profile the entire training loop
    with profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=True
    ) as prof:
        for epoch in range(num_epochs):
            epoch_loss = 0.0

            with record_function(f"epoch_{epoch}"):
                for batch_idx, (inputs, labels) in enumerate(train_data):
                    # Forward pass
                    with record_function("forward_pass"):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    # Backward pass
                    with record_function("backward_pass"):
                        optimizer.zero_grad()
                        loss.backward()

                    # Optimizer step
                    with record_function("optimizer_step"):
                        optimizer.step()

                    epoch_loss += loss.item()

            avg_loss = epoch_loss / num_batches
            print(f"Epoch {epoch + 1}/{num_epochs} - Avg Loss: {avg_loss:.4f}")

    # 4. Print the profiling results

    print(prof.key_averages().table(row_limit=20))

    print(prof.key_averages(group_by_input_shape=True).table(row_limit=15))

    # 5. Analyze time spent in each phase

    events = prof.key_averages()

    print(events)

    for evt in events:
        if "backward_pass" in evt.key:
            print(evt)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    profile_training()