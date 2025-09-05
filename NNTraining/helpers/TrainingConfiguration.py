from dataclasses import dataclass
from typing import List, Any
from jaxlib.xla_extension import Array


@dataclass
class TrainingConfiguration:
    learningRate: float
    numEpochs: int
    batchSize: int
    randomKey: Array
    layerSizes: List[int]
