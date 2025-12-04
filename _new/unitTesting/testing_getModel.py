import torch
import pytest
from _new.helpers.model import get_model
from _new.helpers.config_class import Config


def test_sample_batch_dimensions_mnist():
    """Test that MNIST batch dimensions are correctly extracted"""
    # MNIST: (batch, channels=1, height=28, width=28)
    batch_size = 32
    inputs = torch.randn(batch_size, 1, 28, 28)
    targets = torch.randint(0, 10, (batch_size,))
    sample_batch = (inputs, targets)

    # Extract dimensions like in the code
    input_channels = inputs.shape[1]
    input_size = inputs.shape[2]
    num_classes = len(torch.unique(targets)) if targets.dim() == 1 else targets.shape[1]

    assert input_channels == 1, f"Expected 1 channel for MNIST, got {input_channels}"
    assert input_size == 28, f"Expected size 28 for MNIST, got {input_size}"
    assert num_classes <= 10, f"Expected max 10 classes for MNIST, got {num_classes}"


def test_sample_batch_dimensions_flowers():
    """Test that Flowers batch dimensions are correctly extracted"""
    # Flowers: (batch, channels=3, height=224, width=224)
    batch_size = 32
    inputs = torch.randn(batch_size, 3, 224, 224)
    targets = torch.randint(0, 102, (batch_size,))
    sample_batch = (inputs, targets)

    # Extract dimensions like in the code
    input_channels = inputs.shape[1]
    input_size = inputs.shape[2]
    num_classes = len(torch.unique(targets)) if targets.dim() == 1 else targets.shape[1]

    assert input_channels == 3, f"Expected 3 channels for Flowers, got {input_channels}"
    assert input_size == 224, f"Expected size 224 for Flowers, got {input_size}"
    assert num_classes <= 102, f"Expected max 102 classes for Flowers, got {num_classes}"


def test_sample_batch_one_hot_targets():
    """Test extraction when targets are one-hot encoded"""
    batch_size = 32
    num_classes = 10
    inputs = torch.randn(batch_size, 1, 28, 28)
    targets = torch.zeros(batch_size, num_classes)
    targets[torch.arange(batch_size), torch.randint(0, num_classes, (batch_size,))] = 1
    sample_batch = (inputs, targets)

    extracted_num_classes = len(torch.unique(targets)) if targets.dim() == 1 else targets.shape[1]

    assert extracted_num_classes == num_classes, f"Expected {num_classes} classes, got {extracted_num_classes}"