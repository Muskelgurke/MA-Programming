import pytest
import torch
import torch.nn as nn
from torchvision import models
from _new.helpers.model import get_adapted_model


class TestGetAdaptedModel:
    """Test-Suite für get_adapted_model mit verschiedenen Architekturen."""

    # Test-Specs für verschiedene Datensätze
    SPECS_MNIST = {
        "input_channels": 1,
        "input_size": 28,
        "num_classes": 10
    }

    SPECS_CIFAR10 = {
        "input_channels": 3,
        "input_size": 32,
        "num_classes": 10
    }

    SPECS_CIFAR100 = {
        "input_channels": 3,
        "input_size": 32,
        "num_classes": 100
    }

    SPECS_FLOWERS102 = {
        "input_channels": 3,
        "input_size": 224,
        "num_classes": 102
    }

    @pytest.mark.parametrize("model_fn,specs,expected_in_channels,expected_out_features", [
        # MNIST Tests (1 Kanal)
        (models.alexnet, SPECS_MNIST, 1, 10),
        (models.vgg16, SPECS_MNIST, 1, 10),
        (models.resnet18, SPECS_MNIST, 1, 10),
        (models.resnet50, SPECS_MNIST, 1, 10),
        (models.densenet121, SPECS_MNIST, 1, 10),
        (models.mobilenet_v3_small, SPECS_MNIST, 1, 10),
        (models.efficientnet_v2_s, SPECS_MNIST, 1, 10),

        # CIFAR-10 Tests (3 Kanäle, 10 Klassen)
        (models.alexnet, SPECS_CIFAR10, 3, 10),
        (models.vgg16, SPECS_CIFAR10, 3, 10),
        (models.resnet18, SPECS_CIFAR10, 3, 10),
        (models.resnet50, SPECS_CIFAR10, 3, 10),

        # CIFAR-100 Tests (3 Kanäle, 100 Klassen)
        (models.alexnet, SPECS_CIFAR100, 3, 100),
        (models.resnet18, SPECS_CIFAR100, 3, 100),

        # Flowers102 Tests (3 Kanäle, 102 Klassen)
        (models.alexnet, SPECS_FLOWERS102, 3, 102),
        (models.vgg16, SPECS_FLOWERS102, 3, 102),
    ])
    def test_input_output_adaptation(self, model_fn, specs, expected_in_channels, expected_out_features):
        """Testet, ob Input- und Output-Layer korrekt angepasst werden."""
        model = get_adapted_model(model_fn, specs)

        # Test Input-Channels
        actual_in_channels = self._get_first_conv_in_channels(model)
        assert actual_in_channels == expected_in_channels, \
            f"{model_fn.__name__}: Expected {expected_in_channels} input channels, got {actual_in_channels}"

        # Test Output-Features
        actual_out_features = self._get_last_linear_out_features(model)
        assert actual_out_features == expected_out_features, \
            f"{model_fn.__name__}: Expected {expected_out_features} output features, got {actual_out_features}"

    def test_alexnet_mnist(self):
        """Detaillierter Test für AlexNet mit MNIST."""
        model = get_adapted_model(models.alexnet, self.SPECS_MNIST)

        # Input-Layer prüfen
        first_conv = model.features[0]
        assert isinstance(first_conv, nn.Conv2d)
        assert first_conv.in_channels == 1
        assert first_conv.out_channels == 64

        # Output-Layer prüfen
        last_linear = model.classifier[-1]
        assert isinstance(last_linear, nn.Linear)
        assert last_linear.out_features == 10

        # Forward-Pass testen
        dummy_input = torch.randn(2, 1, 84, 84)
        output = model(dummy_input)
        assert output.shape == (2, 10)

    def test_vgg16_mnist(self):
        """Detaillierter Test für VGG16 mit MNIST."""
        model = get_adapted_model(models.vgg16, self.SPECS_MNIST)

        # Input-Layer prüfen
        first_conv = model.features[0]
        assert isinstance(first_conv, nn.Conv2d)
        assert first_conv.in_channels == 1
        assert first_conv.out_channels == 64

        # Output-Layer prüfen
        last_linear = model.classifier[-1]
        assert isinstance(last_linear, nn.Linear)
        assert last_linear.out_features == 10

        # Forward-Pass testen
        dummy_input = torch.randn(2, 1, 56, 56)
        output = model(dummy_input)
        assert output.shape == (2, 10)

    def test_resnet18_mnist(self):
        """Detaillierter Test für ResNet18 mit MNIST."""
        model = get_adapted_model(models.resnet18, self.SPECS_MNIST)

        # Input-Layer prüfen
        assert isinstance(model.conv1, nn.Conv2d)
        assert model.conv1.in_channels == 1
        assert model.conv1.out_channels == 64

        # Output-Layer prüfen
        assert isinstance(model.fc, nn.Linear)
        assert model.fc.out_features == 10

        # Forward-Pass testen
        dummy_input = torch.randn(2, 1, 28, 28)
        output = model(dummy_input)
        assert output.shape == (2, 10)

    def test_resnet50_cifar100(self):
        """Detaillierter Test für ResNet50 mit CIFAR-100."""
        model = get_adapted_model(models.resnet50, self.SPECS_CIFAR100)

        # Input-Layer (sollte 3 Kanäle bleiben)
        assert model.conv1.in_channels == 3
        assert model.conv1.out_channels == 64

        # Output-Layer (100 Klassen)
        assert isinstance(model.fc, nn.Linear)
        assert model.fc.out_features == 100

        # Forward-Pass testen
        dummy_input = torch.randn(2, 3, 32, 32)
        output = model(dummy_input)
        assert output.shape == (2, 100)

    def test_densenet121_mnist(self):
        """Detaillierter Test für DenseNet121 mit MNIST."""
        model = get_adapted_model(models.densenet121, self.SPECS_MNIST)

        # Input-Layer prüfen
        first_conv = model.features[0]
        assert isinstance(first_conv, nn.Conv2d)
        assert first_conv.in_channels == 1
        assert first_conv.out_channels == 64

        # Output-Layer prüfen
        assert isinstance(model.classifier, nn.Linear)
        assert model.classifier.out_features == 10

        # Forward-Pass testen
        dummy_input = torch.randn(2, 1, 56, 56)
        output = model(dummy_input)
        assert output.shape == (2, 10)

    def test_mobilenet_v3_mnist(self):
        """Detaillierter Test für MobileNetV3 mit MNIST."""
        model = get_adapted_model(models.mobilenet_v3_small, self.SPECS_MNIST)

        # Input-Layer prüfen
        first_conv = model.features[0][0]
        assert isinstance(first_conv, nn.Conv2d)
        assert first_conv.in_channels == 1

        # Output-Layer prüfen
        assert isinstance(model.classifier, nn.Sequential)
        last_linear = model.classifier[-1]
        assert isinstance(last_linear, nn.Linear)
        assert last_linear.out_features == 10

        # Forward-Pass testen
        dummy_input = torch.randn(2, 1, 28, 28)
        output = model(dummy_input)
        assert output.shape == (2, 10)

    def test_efficientnet_v2_mnist(self):
        """Detaillierter Test für EfficientNetV2 mit MNIST."""
        model = get_adapted_model(models.efficientnet_v2_s, self.SPECS_MNIST)

        # Input-Layer prüfen
        first_conv = model.features[0][0]
        assert isinstance(first_conv, nn.Conv2d)
        assert first_conv.in_channels == 1

        # Output-Layer prüfen
        assert isinstance(model.classifier, nn.Sequential)
        last_linear = model.classifier[-1]
        assert isinstance(last_linear, nn.Linear)
        assert last_linear.out_features == 10

        # Forward-Pass testen
        dummy_input = torch.randn(2, 1, 28, 28)
        output = model(dummy_input)
        assert output.shape == (2, 10)

    def test_no_pretrained_weights(self):
        """Testet, dass keine vortrainierten Gewichte geladen werden."""
        # Erstelle zwei identische Modelle
        model1 = get_adapted_model(models.resnet18, self.SPECS_MNIST)
        model2 = get_adapted_model(models.resnet18, self.SPECS_MNIST)

        # Gewichte sollten unterschiedlich sein (Random Init)
        param1 = next(model1.parameters())
        param2 = next(model2.parameters())

        assert not torch.equal(param1, param2), \
            "Gewichte sind identisch - möglicherweise wurden vortrainierte Gewichte geladen"

    # Helper-Methoden
    def _get_first_conv_in_channels(self, model: nn.Module) -> int:
        """Extrahiert die Anzahl der Input-Channels der ersten Conv-Schicht."""
        if hasattr(model, 'features'):
            # AlexNet, VGG, DenseNet, MobileNet, EfficientNet
            first_layer = model.features[0]
            if isinstance(first_layer, nn.Conv2d):
                return first_layer.in_channels
            elif isinstance(first_layer, nn.Sequential):
                # MobileNet/EfficientNet haben Sequential als ersten Layer
                return first_layer[0].in_channels
        elif hasattr(model, 'conv1'):
            # ResNet
            return model.conv1.in_channels
        raise ValueError("Konnte erste Conv-Schicht nicht finden")

    def _get_last_linear_out_features(self, model: nn.Module) -> int:
        """Extrahiert die Anzahl der Output-Features des letzten Linear-Layers."""
        if hasattr(model, 'classifier'):
            if isinstance(model.classifier, nn.Sequential):
                # AlexNet, VGG, MobileNet, EfficientNet
                return model.classifier[-1].out_features
            elif isinstance(model.classifier, nn.Linear):
                # DenseNet
                return model.classifier.out_features
        elif hasattr(model, 'fc'):
            # ResNet
            return model.fc.out_features
        raise ValueError("Konnte letzten Linear-Layer nicht finden")


# Zusätzliche Edge-Case-Tests
class TestEdgeCases:
    """Tests für Spezialfälle und Fehlererkennung."""

    def test_invalid_specs_missing_keys(self):
        """Testet Verhalten bei fehlenden Spec-Keys."""
        invalid_specs = {"input_channels": 1}  # Fehlt num_classes

        with pytest.raises((KeyError, ValueError)):
            get_adapted_model(models.resnet18, invalid_specs)

    def test_unusual_num_classes(self):
        """Testet mit ungewöhnlichen Klassenanzahlen."""
        specs = {
            "input_channels": 3,
            "input_size": 32,
            "num_classes": 1000  # Viele Klassen
        }

        model = get_adapted_model(models.resnet18, specs)
        assert model.fc.out_features == 1000

    def test_single_class(self):
        """Testet mit nur einer Klasse (Edge-Case)."""
        specs = {
            "input_channels": 1,
            "input_size": 28,
            "num_classes": 1
        }

        model = get_adapted_model(models.resnet18, specs)
        assert model.fc.out_features == 1

        # Forward-Pass
        dummy_input = torch.randn(2, 1, 28, 28)
        output = model(dummy_input)
        assert output.shape == (2, 1)