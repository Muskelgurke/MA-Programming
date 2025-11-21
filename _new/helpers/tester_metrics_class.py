from dataclasses import dataclass, asdict
from typing import List


@dataclass
class TesterMetrics:
    """Dataclass to store training metrics for each epoch"""
    test_loss_per_epoch: float = 0.0
    test_acc_per_epoch: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    def get_csv_fieldnames(self) -> List[str]:
        return list(asdict(self).keys())

    def _trainings_metrics_reset(self) -> None:
        self.test_loss_per_epoch = 0
        self.test_acc_per_epoch = 0