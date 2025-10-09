from dataclasses import dataclass
from typing import Any

@dataclass
class TrainingData:
    training_generator: Any
    train_images: Any
    train_labels: Any
    test_images: Any
    test_labels: Any
    n_targets: int