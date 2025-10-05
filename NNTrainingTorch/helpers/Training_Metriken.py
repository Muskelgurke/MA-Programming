from dataclasses import dataclass
from typing import List
import torch
from werkzeug.formparser import default_stream_factory


@dataclass
class TrainingMetrics:
    """Dataclass to store training metrics for each epoch"""
    avg_train_loss_of_epoch: float = 0.0
    avg_train_acc_of_epoch: float = 0.0
    train_acc: float = 0.0
    accumulated_correct_samples_of_all_batches: int = 0
    accumulated_total_samples_of_all_batches: int = 0
    cosine_similarities_of_estgrad_gradient_for_each_batch: List[float] = None
    avg_cosine_similarity_of_epoch: float = 0.0
    cosine_sim_for_batch: float = 0.0
    estimated_gradients: List[torch.Tensor] = None
    std_of_difference_true_esti_grads: float = 0.0
    std_of_esti_grads: float = 0.0
    std_of_true_grads: float = 0.0
    mse_true_esti_grads: float = 0.0
    mae_true_esti_grad: float = 0.0