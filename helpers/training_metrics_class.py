from dataclasses import dataclass, asdict
from typing import List

@dataclass
class TrainMetrics:
    """Dataclass to store training metrics for each epoch"""
    loss_per_epoch: float = 0.0
    acc_per_epoch: float = 0.0
    num_batches: int = 0
    epoch_duration: float = 0.0
    time_to_converge: float = 0.0

    mem_base_bytes: int = 0
    mem_parameter_bytes: int = 0
    avg_mem_forward_pass_bytes: int = 0
    max_mem_forward_pass_bytes: int = 0
    max_mem_bytes: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    def get_csv_fieldnames(self) -> List[str]:
        return list(asdict(self).keys())

    def clear_train_metrics(self):
        """Reset metrics loss_per_epoch and acc_per_epoch to zero."""
