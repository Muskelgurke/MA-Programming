from dataclasses import dataclass
from typing import List, Any

@dataclass
class TrainingResults:
    train_accs: List[float]
    test_accs: List[float]
    train_losses: List[float]
    test_losses: List[float]
    final_params: Any
    epoch_times: List[float]