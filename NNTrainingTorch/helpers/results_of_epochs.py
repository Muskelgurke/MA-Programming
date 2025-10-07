from dataclasses import dataclass
from typing import List, Any

from NNTrainingTorch.helpers.training_metrics_class import TrainingMetrics


@dataclass
class results_of_epochs:
    train_accs: List[float]
    test_accs: List[float]
    train_losses: List[float]
    test_losses: List[float]
    epoch_times: List[float]
