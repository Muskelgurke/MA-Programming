from dataclasses import dataclass, asdict
from typing import List


@dataclass
class TestMetrics:
    """Dataclass to store training metrics for each epoch"""
    loss_per_epoch: float = 0.0
    acc_per_epoch: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    def get_csv_fieldnames(self) -> List[str]:
        return list(asdict(self).keys())

    def clear_test_metrics(self) -> None:
        """Reset metrics loss_per_epoch and acc_per_epoch to zero."""
        self.loss_per_epoch = 0
        self.acc_per_epoch = 0