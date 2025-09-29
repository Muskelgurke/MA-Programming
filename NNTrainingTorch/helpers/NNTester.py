import torch
from torch.nn import MSELoss

from NNTrainingTorch.helpers.config_class import Config


class Tester:
    def __init__(self,
                 ConfigFile: Config,
                 Model: torch.nn.Module,
                 Test_loader: torch.utils.data.DataLoader,
                 Loss_function: torch.nn.Module,
                 Device: torch.device,
                 Num_epochs: int,
                 Random_seed: int):

        self.config = ConfigFile
        self.model = Model
        self.test_loader = Test_loader
        self.loss_function = Loss_function
        self.device = Device
        self.num_epochs = Num_epochs
        self.random_seed = Random_seed
        self.val_loss = 0
        self.test_acc = 0
        self.correct = 0
        self.total = 0

         # Set model to evaluation mode

    def test_epoch(self):
        # print("Evaluation of Epoch on Test Dataset")
        self.model.eval()
        self.val_loss = 0
        self.test_acc = 0

        if self.config.model_type == "demo_linear_regression":
            self.eval_linearRegression()

        else:
            self.eval_classification()

        return self.val_loss, self.test_acc

    def eval_classification(self):
        with torch.no_grad():  # Disable gradient computation for efficiency
            for data, targets in self.test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                test_loss = self.loss_function(outputs, targets)
                self.val_loss += test_loss.item() * data.size(0)
                _, predicted = torch.max(outputs, 1)
                self.total += targets.size(0)
                self.correct += (predicted == targets).sum().item()
        self.val_loss /= self.total
        self.test_acc = self.correct / self.total

        return self.val_loss, self.test_acc

    def eval_linearRegression(self) -> tuple[float, float]:
        with torch.no_grad():  # Disable gradient computation for efficiency
            for data, targets in self.test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                test_loss = self.loss_function(outputs, targets)
                self.val_loss += test_loss.item() * data.size(0)
                self.total += data.size(0)

        self.val_loss /= self.total
        self.test_acc = 0 # Vielleicht R^2 Score implementieren?

        return self.val_loss, self.test_acc


