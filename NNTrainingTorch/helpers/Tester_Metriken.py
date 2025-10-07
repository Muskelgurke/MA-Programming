from dataclasses import dataclass, asdict, field
from typing import List
import torch
from werkzeug.formparser import default_stream_factory


@dataclass
class TesterMetrics:
    """Dataclass to store training metrics for each epoch"""
    test_loss_per_epoch: float
    test_acc_per_epoch: float

    def to_dict(self) -> dict:
        return asdict(self)

    def get_csv_fieldnames(self) -> List[str]:
        return list(asdict(self).keys())

    def _trainings_metrics_reset(self) -> None:
        self.test_loss_per_epoch.clear()
        self.test_acc_per_epoch.clear()