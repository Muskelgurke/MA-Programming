import torch

class MemoryRecorder:
    def __init__(self):
        """ torch.cuda.memory_allocated() misst in bytes
        """
        self.reset()

    def reset_cuda(self):
        torch.cuda.reset_peak_memory_stats()
        self.memory_stats = []

    def log(self, step_name=""):

        current = torch.cuda.memory_allocated() / 1024 ** 2
        peak = torch.cuda.max_memory_allocated() / 1024 ** 2

        self.memory_stats.append({
            "step": step_name,
            "current_mb": current,
            "peak_mb": peak
        })
        return peak
