from dataclasses import dataclass, asdict, field
from typing import List
import torch
from werkzeug.formparser import default_stream_factory


@dataclass
class TrainingMetrics:
    """Dataclass to store training metrics for each epoch"""
    epoch_avg_train_loss: float = 0.0
    train_acc_of_epoch: float = 0.0
    cosine_of_esti_true_grads_for_each_batch: List[float] = field(default_factory=list)
    avg_cosine_similarity_of_epoch: float = 0.0
    estimated_gradients: List[torch.Tensor] = field(default_factory=list)
    abs_of_diff_true_esti_grads: List[torch.Tensor] = field(default_factory=list)
    std_of_difference_true_esti_grads: List[float] = field(default_factory=list)
    std_of_esti_grads: List[float] = field(default_factory=list)
    std_of_true_grads: List[float] = field(default_factory=list)
    mse_true_esti_grads: List[float] = field(default_factory=list)
    mae_true_esti_grad: List[float] = field(default_factory=list)

    def _to_dict(self) -> dict:
        return asdict(self)

    def _trainings_metrics_reset(self) -> None:
        self.epoch_avg_train_loss = 0.0
        self.train_acc_of_epoch = 0.0
        self.cosine_of_esti_true_grads_for_each_batch.clear()
        self.avg_cosine_similarity_of_epoch = 0.0
        self.estimated_gradients.clear()
        self.abs_of_diff_true_esti_grads.clear()
        self.std_of_difference_true_esti_grads.clear()
        self.std_of_esti_grads.clear()
        self.std_of_true_grads.clear()
        self.mse_true_esti_grads.clear()
        self.mae_true_esti_grad.clear()ss