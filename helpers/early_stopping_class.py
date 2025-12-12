import torch
from typing import Optional, Dict, Any


class EarlyStopping:

    def __init__(self, patience: int, delta: float, max_loss_threshold: float = 10000):
        self.patience = patience
        self.delta = delta
        self.max_loss_threshold = max_loss_threshold
        self.best_score: Optional[float] = None
        self.counter = 0
        self.early_stop = False
        self.best_model_state: Optional[Dict] = None
        self.stop_info: Dict[str, Any] = {}

    def check_and_update(
            self,
            train_loss: float,
            val_loss: float,
            model: torch.nn.Module,
            epoch: int
    ) -> None:
        """
        Prüft Train- und Validation-Loss und entscheidet über Early Stop
        Setzt self.should_stop = True wenn Training gestoppt werden soll
        """
        # 1. Prüfe auf NaN oder explodierte Loss in Train Loss
        if self._check_invalid_train_loss(train_loss, epoch):
            return

        # 2. Prüfe auf ungültige Validation Loss
        if self._check_invalid_val_loss(val_loss, epoch):
            return

        # 3. Normale Patience-Logik
        self._check_patience(val_loss, model, epoch)

    def _check_invalid_train_loss(self, train_loss: float, epoch: int) -> bool:
        """Prüft ob Train Loss NaN ist oder Schwellenwert überschreitet"""
        if train_loss is None or torch.isnan(torch.tensor(train_loss)):
            self.early_stop = True
            self.stop_info = {
                "reason": "nan_train_loss",
                "stopped_at_epoch": epoch,
                "train_loss": str(train_loss)
            }
            return True

        if train_loss > self.max_loss_threshold:
            self.early_stop = True
            self.stop_info = {
                "reason": "train_loss_exploded",
                "stopped_at_epoch": epoch,
                "train_loss": train_loss,
                "threshold": self.max_loss_threshold
            }
            return True

        return False

    def _check_invalid_val_loss(self, val_loss: float, epoch: int) -> bool:
        """Prüft ob Validation Loss ungültig ist (NaN oder explodiert)"""
        if val_loss is None or torch.isnan(torch.tensor(val_loss)):
            self.early_stop = True
            self.stop_info = {
                "reason": "nan_val_loss",
                "stopped_at_epoch": epoch,
                "val_loss": str(val_loss)
            }
            return True

        if val_loss > self.max_loss_threshold:
            self.early_stop = True
            self.stop_info = {
                "reason": "val_loss_exploded",
                "stopped_at_epoch": epoch,
                "val_loss": val_loss,
                "threshold": self.max_loss_threshold
            }
            return True

        return False

    def _check_patience(self, val_loss: float, model: torch.nn.Module, epoch: int) -> None:
        """Normale Patience-basierte Early Stopping Logik"""
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return

        threshold = self.best_score + self.delta

        if score <= threshold:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                self.stop_info = {
                    "reason": "patience_exceeded",
                    "stopped_at_epoch": epoch,
                    "patience": self.patience,
                    "best_val_loss": -self.best_score
                }
        else:
            self.best_score = score
            self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0

    def load_best_model(self, model: torch.nn.Module) -> None:
        """Lädt das beste gespeicherte Modell"""
        if self.best_model_state is not None:
            device = next(model.parameters()).device
            state_dict = {k: v.to(device) for k, v in self.best_model_state.items()}
            model.load_state_dict(state_dict)

    def get_stop_info(self) -> Dict[str, Any]:
        """Gibt Informationen über den Stop-Grund zurück"""
        return self.stop_info

    def reset(self) -> None:
        """Setzt alle Zustände zurück"""
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.best_model_state = None
        self.stop_info = {}

