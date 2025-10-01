import torch
from torch.nn import MSELoss
from tqdm import tqdm

from NNTrainingTorch.helpers.config_class import Config


class Tester:
    def __init__(self,
                 configFile: Config,
                 model: torch.nn.Module,
                 test_loader: torch.utils.data.DataLoader,
                 loss_function: torch.nn.Module,
                 device: torch.device,
                 total_epochs: int,
                 random_seed: int):

        self.config = configFile
        self.model = model
        self.test_loader = test_loader
        self.loss_function = loss_function
        self.device = device
        self.total_epochs = total_epochs
        self.random_seed = random_seed
        self.val_loss = 0
        self.test_acc = 0
        self.correct = 0
        self.total = 0

         # Set model to evaluation mode

    def test_epoch(self, epoch: int) -> tuple[float, float]:
        self.num_epochs=epoch
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
            pbar = tqdm(self.test_loader, desc=f'Test Epoch {self.num_epochs}/{self.total_epochs}')
            for data, targets in self.test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)

                test_loss = self.loss_function(outputs, targets)


                self.val_loss += test_loss.item() * data.size(0)
                _, predicted = torch.max(outputs.data, 1)
                self.total += targets.size(0)
                self.correct += (predicted == targets).sum().item()

                pbar.update(1)
                pbar.set_postfix({
                    'Loss': f'{test_loss.item():.4f}',
                    'Acc': f'{(self.correct / self.total) * 100:.4f}%'
                })

            pbar.close()
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


