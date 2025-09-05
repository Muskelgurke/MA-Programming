from dataclasses import dataclass
from typing import List, Any

@dataclass
class TrainingResults:
    train_accs: List[float]
    test_accs: List[float]
    train_loss: List[float]
    test_loss: List[float]
    final_params: Any
    epoch_times: List[float]