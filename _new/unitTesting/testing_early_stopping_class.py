import unittest
import torch
import torch.nn as nn
from _new.helpers.early_stopping_class import EarlyStopping


class SimpleModel(nn.Module):
    """Einfaches Test-Modell"""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


class TestEarlyStopping(unittest.TestCase):

    def setUp(self):
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
        self.assertEqual(self.early_stopping.patience, 3)
        self.assertEqual(self.early_stopping.delta, 0.001)
        self.assertEqual(self.early_stopping.max_loss_threshold, 10000.0)
        self.assertIsNone(self.early_stopping.best_score)
        self.assertEqual(self.early_stopping.counter, 0)
        self.assertFalse(self.early_stopping.should_stop)
        self.assertIsNone(self.early_stopping.best_model_state)
        self.assertEqual(self.early_stopping.stop_info, {})

    def test_nan_train_loss_detection(self):
        """Test: NaN in Train Loss wird erkannt"""
        self.early_stopping.check_and_update(
            train_loss=float('nan'),
            val_loss=0.5,
            model=self.model,
            epoch=0
        )

        self.assertTrue(self.early_stopping.should_stop)
        self.assertEqual(self.early_stopping.stop_info['reason'], 'nan_train_loss')
        self.assertEqual(self.early_stopping.stop_info['stopped_at_epoch'], 0)

    def test_nan_val_loss_detection(self):
        """Test: NaN in Validation Loss wird erkannt"""
        self.early_stopping.check_and_update(
            train_loss=0.5,
            val_loss=float('nan'),
            model=self.model,
            epoch=2
        )

        self.assertTrue(self.early_stopping.should_stop)
        self.assertEqual(self.early_stopping.stop_info['reason'], 'nan_val_loss')
        self.assertEqual(self.early_stopping.stop_info['stopped_at_epoch'], 2)

    def test_train_loss_exploded(self):
        """Test: Explodierte Train Loss wird erkannt"""
        self.early_stopping.check_and_update(
            train_loss=15000.0,
            val_loss=0.5,
            model=self.model,
            epoch=5
        )

        self.assertTrue(self.early_stopping.should_stop)
        self.assertEqual(self.early_stopping.stop_info['reason'], 'train_loss_exploded')
        self.assertEqual(self.early_stopping.stop_info['train_loss'], 15000.0)
        self.assertEqual(self.early_stopping.stop_info['threshold'], 10000.0)

    def test_val_loss_exploded(self):
        """Test: Explodierte Validation Loss wird erkannt"""
        self.early_stopping.check_and_update(
            train_loss=0.5,
            val_loss=20000.0,
            model=self.model,
            epoch=3
        )

        self.assertTrue(self.early_stopping.should_stop)
        self.assertEqual(self.early_stopping.stop_info['reason'], 'val_loss_exploded')
        self.assertEqual(self.early_stopping.stop_info['val_loss'], 20000.0)

    def test_patience_exceeded(self):
        """Test: Patience wird nach mehreren Epochen ohne Verbesserung überschritten"""
        # Erste Epoch: Beste Loss
        self.early_stopping.check_and_update(1.0, 1.0, self.model, 0)
        self.assertFalse(self.early_stopping.should_stop)
        self.assertEqual(self.early_stopping.counter, 0)

        # Weitere Epochen ohne Verbesserung
        for epoch in range(1, 4):
            self.early_stopping.check_and_update(1.0, 1.0, self.model, epoch)

        self.assertTrue(self.early_stopping.should_stop)
        self.assertEqual(self.early_stopping.stop_info['reason'], 'patience_exceeded')
        self.assertEqual(self.early_stopping.counter, 3)

    def test_patience_reset_on_improvement(self):
        """Test: Counter wird bei Verbesserung zurückgesetzt"""
        # Epoch 0: Initiale Loss
        self.early_stopping.check_and_update(1.0, 1.0, self.model, 0)

        # Epoch 1-2: Keine Verbesserung
        self.early_stopping.check_and_update(1.0, 1.0, self.model, 1)
        self.early_stopping.check_and_update(1.0, 1.0, self.model, 2)
        self.assertEqual(self.early_stopping.counter, 2)

        # Epoch 3: Verbesserung
        self.early_stopping.check_and_update(0.5, 0.5, self.model, 3)
        self.assertEqual(self.early_stopping.counter, 0)
        self.assertFalse(self.early_stopping.should_stop)

    def test_best_model_state_saved(self):
        """Test: Bestes Modell wird gespeichert"""
        # Erste Epoch
        self.early_stopping.check_and_update(0.8, 0.8, self.model, 0)
        self.assertIsNotNone(self.early_stopping.best_model_state)
        first_state = self.early_stopping.best_model_state

        # Epoch mit Verbesserung
        self.early_stopping.check_and_update(0.5, 0.5, self.model, 1)
        second_state = self.early_stopping.best_model_state

        # States sollten unterschiedlich sein
        self.assertIsNot(first_state, second_state)

    def test_load_best_model(self):
        """Test: Bestes Modell kann geladen werden"""
        # Speichere initialen State
        initial_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        # Erste Epoch
        self.early_stopping.check_and_update(0.5, 0.5, self.model, 0)

        # Verändere Modell-Parameter
        for param in self.model.parameters():
            param.data.fill_(999.0)

        # Lade bestes Modell zurück
        self.early_stopping.load_best_model(self.model)

        # Prüfe ob State wiederhergestellt wurde
        current_state = self.model.state_dict()
        for key in initial_state:
            # Werte sollten ähnlich sein (nicht genau gleich wegen Rounding)
            self.assertTrue(torch.allclose(current_state[key], initial_state[key], atol=1e-5))

    def test_get_stop_info_when_not_stopped(self):
        """Test: Stop Info ist leer wenn nicht gestoppt"""
        self.assertEqual(self.early_stopping.get_stop_info(), {})

    def test_get_stop_info_after_stop(self):
        """Test: Stop Info enthält Informationen nach Stop"""
        self.early_stopping.check_and_update(
            train_loss=float('nan'),
            val_loss=0.5,
            model=self.model,
            epoch=10
        )

        info = self.early_stopping.get_stop_info()
        self.assertIn('reason', info)
        self.assertIn('stopped_at_epoch', info)
        self.assertEqual(info['stopped_at_epoch'], 10)

    def test_reset(self):
        """Test: Reset setzt alle Werte zurück"""
        # Triggere Stop
        self.early_stopping.check_and_update(
            train_loss=float('nan'),
            val_loss=0.5,
            model=self.model,
            epoch=5
        )

        # Reset
        self.early_stopping.reset()

        # Prüfe Rücksetzen
        self.assertIsNone(self.early_stopping.best_score)
        self.assertEqual(self.early_stopping.counter, 0)
        self.assertFalse(self.early_stopping.should_stop)
        self.assertIsNone(self.early_stopping.best_model_state)
        self.assertEqual(self.early_stopping.stop_info, {})

    def test_delta_threshold(self):
        """Test: Delta-Schwellenwert wird korrekt angewendet"""
        # Erste Epoch
        self.early_stopping.check_and_update(1.0, 1.0, self.model, 0)

        # Kleine Verbesserung unter Delta
        self.early_stopping.check_and_update(0.999, 0.9995, self.model, 1)
        self.assertEqual(self.early_stopping.counter, 1)

        # Verbesserung über Delta
        self.early_stopping.check_and_update(0.95, 0.95, self.model, 2)
        self.assertEqual(self.early_stopping.counter, 0)

    def test_none_train_loss(self):
        """Test: None als Train Loss wird erkannt"""
        self.early_stopping.check_and_update(
            train_loss=None,
            val_loss=0.5,
            model=self.model,
            epoch=1
        )

        self.assertTrue(self.early_stopping.should_stop)
        self.assertEqual(self.early_stopping.stop_info['reason'], 'nan_train_loss')

    def test_none_val_loss(self):
        """Test: None als Validation Loss wird erkannt"""
        self.early_stopping.check_and_update(
            train_loss=0.5,
            val_loss=None,
            model=self.model,
            epoch=1
        )

        self.assertTrue(self.early_stopping.should_stop)
        self.assertEqual(self.early_stopping.stop_info['reason'], 'nan_val_loss')

    def test_model_on_gpu_if_available(self):
        """Test: Modell auf GPU wenn verfügbar"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA nicht verfügbar")

        device = torch.device('cuda')
        model = SimpleModel().to(device)
        early_stopping = EarlyStopping(patience=3, delta=0.001)

        # Update mit GPU-Modell
        early_stopping.check_and_update(0.5, 0.5, model, 0)

        # Best Model State sollte auf CPU sein
        for tensor in early_stopping.best_model_state.values():
            self.assertFalse(tensor.is_cuda)

        # Load Best Model sollte zurück auf GPU gehen
        early_stopping.load_best_model(model)
        for param in model.parameters():
            self.assertTrue(param.is_cuda)

    def test_consecutive_improvements(self):
        """Test: Mehrere aufeinanderfolgende Verbesserungen"""
        losses = [1.0, 0.9, 0.8, 0.7, 0.6]

        for epoch, loss in enumerate(losses):
            self.early_stopping.check_and_update(loss, loss, self.model, epoch)
            self.assertFalse(self.early_stopping.should_stop)
            self.assertEqual(self.early_stopping.counter, 0)

        self.assertEqual(self.early_stopping.best_score, -0.6)


if __name__ == '__main__':
    unittest.main()