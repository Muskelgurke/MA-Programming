import torch

class AnalyticalMemoryBP:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.total_elements = 0
        self.mem_of_model = 0


    def memory_usage(self) -> int:
        for layer in self.model.modules():
            if isinstance(layer, torch.nn.Linear):
                self.calculate_for_linear(layer)
            if isinstance(layer, torch.nn.Conv1d):
                print("calc conv1d not implemented yet")
                pass
            if isinstance(layer, torch.nn.Conv2d):
                print("calc conv2d not implemented yet")
                pass
            if isinstance(layer, torch.nn.Conv3d):
                print("calc conv3d not implemented yet")
                pass




    def calculate_for_linear(self,l: torch.nn.Linear) -> int:
        num_weights = l.in_features * l.out_features
        self.total_elements += num_weights



if __name__ == "__main__":
    model = torch.nn.Linear(10,10)
    analytical = AnalyticalMemoryBP(model)
    analytical.memory_usage()
    print(f"{analytical.mem_of_model}")


