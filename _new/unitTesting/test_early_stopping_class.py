import pytest
import torch
import torch.nn as nn
from helpers.early_stopping_class import EarlyStopping


class SimpleModel(nn.Module):
    """Einfaches Test-Modell"""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)

class TestEarlyStopping():

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup vor jedem Test"""
        self.model = SimpleModel()
        self.patience = 3
        self.delta = 0.001
        self.max_loss_threshold = 10000.0
        self.early_stopping = EarlyStopping(
            patience=self.patience,
            delta=self.delta,
            max_loss_threshold=self.max_loss_threshold
        )

    def test_initialization(self):
        """Test: Korrekte Initialisierung"""
        assert self.early_stopping.patience == 3
        assert self.early_stopping.delta == 0.001
        assert self.early_stopping.max_loss_threshold == 10000.0
        assert self.early_stopping.best_score is None
        assert self.early_stopping.counter == 0
        assert self.early_stopping.early_stop is False
        assert self.early_stopping.best_model_state is None
        assert self.early_stopping.stop_info == {}

    def test_nan_train_loss_detection(self):
        """Test: NaN in Train Loss wird erkannt"""
        self.early_stopping.check_and_update(
            train_loss=float('nan'),
            val_loss=0.5,
            model=self.model,
            epoch=0
        )

        assert self.early_stopping.early_stop is True
        assert self.early_stopping.stop_info['reason'] == 'nan_train_loss'
        assert self.early_stopping.stop_info['stopped_at_epoch'] == 0

    def test_nan_val_loss_detection(self):
        """Test: NaN in Validation Loss wird erkannt"""
        self.early_stopping.check_and_update(
            train_loss=0.5,
            val_loss=float('nan'),
            model=self.model,
            epoch=2
        )

        assert self.early_stopping.early_stop is True
        assert self.early_stopping.stop_info['reason'] == 'nan_val_loss'
        assert self.early_stopping.stop_info['stopped_at_epoch'] == 2

    def test_train_loss_exploded(self):
        """Test: Explodierte Train Loss wird erkannt"""
        self.early_stopping.check_and_update(
            train_loss=15000.0,
            val_loss=0.5,
            model=self.model,
            epoch=5
        )

        assert self.early_stopping.early_stop is True
        assert self.early_stopping.stop_info['reason'] == 'train_loss_exploded'
        assert self.early_stopping.stop_info['train_loss'] == 15000.0
        assert self.early_stopping.stop_info['threshold'] == 10000.0

    def test_val_loss_exploded(self):
        """Test: Explodierte Validation Loss wird erkannt"""
        self.early_stopping.check_and_update(
            train_loss=0.5,
            val_loss=20000.0,
            model=self.model,
            epoch=3
        )

        assert self.early_stopping.early_stop is True
        assert self.early_stopping.stop_info['reason'] == 'val_loss_exploded'
        assert self.early_stopping.stop_info['val_loss'] == 20000.0

    def test_patience_exceeded(self):
        """Test: Patience wird nach mehreren Epochen ohne Verbesserung überschritten"""
        self.early_stopping.check_and_update(1.0, 1.0, self.model, 0)
        assert self.early_stopping.early_stop is False
        assert self.early_stopping.counter == 0

        for epoch in range(1, 4):
            self.early_stopping.check_and_update(1.0, 1.0, self.model, epoch)

        assert self.early_stopping.early_stop is True
        assert self.early_stopping.stop_info['reason'] == 'patience_exceeded'
        assert self.early_stopping.counter == 3

    def test_patience_reset_on_improvement(self):
        """Test: Counter wird bei Verbesserung zurückgesetzt"""
        self.early_stopping.check_and_update(1.0, 1.0, self.model, 0)

        self.early_stopping.check_and_update(1.0, 1.0, self.model, 1)
        self.early_stopping.check_and_update(1.0, 1.0, self.model, 2)
        assert self.early_stopping.counter == 2

        self.early_stopping.check_and_update(0.5, 0.5, self.model, 3)
        assert self.early_stopping.counter == 0
        assert self.early_stopping.early_stop is False

    def test_best_model_state_saved(self):
        """Test: Bestes Modell wird gespeichert"""
        self.early_stopping.check_and_update(0.8, 0.8, self.model, 0)
        assert self.early_stopping.best_model_state is not None
        first_state = self.early_stopping.best_model_state

        self.early_stopping.check_and_update(0.5, 0.5, self.model, 1)
        second_state = self.early_stopping.best_model_state

        assert first_state is not second_state

    def test_load_best_model(self):
        """Test: Bestes Modell kann geladen werden"""
        initial_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        self.early_stopping.check_and_update(0.5, 0.5, self.model, 0)

        for param in self.model.parameters():
            param.data.fill_(999.0)

        self.early_stopping.load_best_model(self.model)

        current_state = self.model.state_dict()
        for key in initial_state:
            assert torch.allclose(current_state[key], initial_state[key], atol=1e-5)

    def test_get_stop_info_when_not_stopped(self):
        """Test: Stop Info ist leer wenn nicht gestoppt"""
        assert self.early_stopping.get_stop_info() == {}

    def test_get_stop_info_after_stop(self):
        """Test: Stop Info enthält Informationen nach Stop"""
        self.early_stopping.check_and_update(
            train_loss=float('nan'),
            val_loss=0.5,
            model=self.model,
            epoch=10
        )

        info = self.early_stopping.get_stop_info()
        assert 'reason' in info
        assert 'stopped_at_epoch' in info
        assert info['stopped_at_epoch'] == 10

    def test_reset(self):
        """Test: Reset setzt alle Werte zurück"""
        self.early_stopping.check_and_update(
            train_loss=float('nan'),
            val_loss=0.5,
            model=self.model,
            epoch=5
        )

        self.early_stopping.reset()

        assert self.early_stopping.best_score is None
        assert self.early_stopping.counter == 0
        assert self.early_stopping.early_stop is False
        assert self.early_stopping.best_model_state is None
        assert self.early_stopping.stop_info == {}

    def test_delta_threshold(self):
        """Test: Delta-Schwellenwert wird korrekt angewendet"""
        self.early_stopping.check_and_update(1.0, 1.0, self.model, 0)

        self.early_stopping.check_and_update(0.999, 0.9995, self.model, 1)
        assert self.early_stopping.counter == 1

        self.early_stopping.check_and_update(0.95, 0.95, self.model, 2)
        assert self.early_stopping.counter == 0

    def test_none_train_loss(self):
        """Test: None als Train Loss wird erkannt"""
        self.early_stopping.check_and_update(
            train_loss=None,
            val_loss=0.5,
            model=self.model,
            epoch=1
        )

        assert self.early_stopping.early_stop is True
        assert self.early_stopping.stop_info['reason'] == 'nan_train_loss'

    def test_none_val_loss(self):
        """Test: None als Validation Loss wird erkannt"""
        self.early_stopping.check_and_update(
            train_loss=0.5,
            val_loss=None,
            model=self.model,
            epoch=1
        )

        assert self.early_stopping.early_stop is True
        assert self.early_stopping.stop_info['reason'] == 'nan_val_loss'

    def test_model_on_gpu_if_available(self):
        """Test: Modell auf GPU wenn verfügbar"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA nicht verfügbar")

        device = torch.device('cuda')
        model = SimpleModel().to(device)
        early_stopping = EarlyStopping(patience=3, delta=0.001)

        early_stopping.check_and_update(0.5, 0.5, model, 0)

        for tensor in early_stopping.best_model_state.values():
            assert tensor.is_cuda is False

        early_stopping.load_best_model(model)
        for param in model.parameters():
            assert param.is_cuda is True

    def test_consecutive_improvements(self):
        """Test: Mehrere aufeinanderfolgende Verbesserungen"""
        losses = [1.0, 0.9, 0.8, 0.7, 0.6]

        for epoch, loss in enumerate(losses):
            self.early_stopping.check_and_update(loss, loss, self.model, epoch)
            assert self.early_stopping.early_stop is False
            assert self.early_stopping.counter == 0

        assert self.early_stopping.best_score == -0.6
