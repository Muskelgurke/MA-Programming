from NNTraining.helpers import saving
from NNTraining.helpers.TrainingConfiguration import TrainingConfiguration
from NNTraining.helpers.TrainingResults import TrainingResults
from NNTraining.helpers.TrainingData import TrainingData

from typing import List, Tuple, Callable
from torch.utils.data import DataLoader, default_collate

import numpy as np
import time
import yaml
import torchvision.datasets as datasets

import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax.scipy.special import logsumexp
from jax import Array
from jax.tree_util import tree_map

# We need a fuction to intilize the weights and biases for a dense neural network layer
def random_layer_params(m: int, n: int,key: int,scale: float=1e-2) -> tuple[Array, Array]:
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n,m)), scale * random.normal(b_key, (n,))

# initialize all layers for a fully-connected NN with sizes
# sizes = number of neurons in each layer

def init_network_params(sizes: List[int], key: int) -> List[Tuple[Array, Array]]:
  keys = random.split(key, len(sizes))
  return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

def relu(x: Array) -> Array:
  return jnp.maximum(0, x)

def forward(params: List[Tuple[Array, Array]], image: Array) -> Array:
  """ Predict the class of a single Neuronal Layer"""

  activations = image # set input to the first layer
  for weights, bias in params[:-1]:
    outputs = jnp.dot(weights, activations) + bias # Neuronen berechnung y = w * x + b
    activations = relu(outputs) # aktiviert die Neuronen mit ReLu und setzt die Ausgabe als Input für die nächste Schicht

  final_w, final_b = params[-1]
  logits = jnp.dot(final_w, activations) + final_b
  return logits - logsumexp(logits) # Bessere und numerisch stabilere Softmax

def buildAutoBatchedFktForJAX():
    global batched_predict
    batched_predict = vmap(forward, in_axes=(None, 0))

def one_hot(x: np.ndarray, k: int, dtype=jnp.float32) -> Array:
  """Create a one-hot encoding of x of size k.
  1.Klassen-Indizes -> jnp.arange(k)
  2.x[:,None] -> x in Spaltenvektor umwandeln
  3.Boolschen werte  in den vorgegebenen dtype umwandeln
  """
  return jnp.array(x[:, None] == jnp.arange(k), dtype)

def accuracy(params: List[Tuple[Array, Array]], images: Array, targets:Array)-> Array:
  """
  Nimmt die batched_predict Funktion, um Vorhersagen für alle Bilder im Batch zu machen,
  und vergleicht diese Vorhersagen mit den tatsächlichen Zielklassen.
  Ein Wert zwischen 0.0 (0% korrekt) und 1.0 (100% korrekt), der angibt, wie gut das Netzwerk die Testdaten klassifiziert."""

  target_class = jnp.argmax(targets, axis=1)
  predicted_class = jnp.argmax(batched_predict(params,images), axis=1)
  return jnp.mean(predicted_class == target_class)

def loss_fn(params: List[Tuple[Array, Array]], images:Array, targets:Array)-> Array:
  """ Negative Log-Likelihood(NLL) eine Form von Cross-Entropy-Loss"""
  preds = batched_predict(params,images)
  return -jnp.mean(preds*targets)

def grad_calculator_backward(params: List[Tuple[Array, Array]], x: Array, y: Array)-> Array:
  return grad(loss_fn)(params, x, y)


@jit
def update(params: List[Tuple[Array,Array]], x: Array, y: Array, learningRate: float, grad_calculator: Callable[[List[Tuple[Array, Array]], Array, Array,], Array])-> List[Tuple[Array,Array]]:
  grads = grad_calculator(params, x, y)
  return [(w - learningRate * dw, b - learningRate * db)
          for (w,b), (dw,db)in zip(params, grads)]

def numpy_collate(batch: List)-> List:
  """  Collate function specifies how to combine a list of data samples into a batch.
  default_collate creates pytorch tensors, then tree_map converts them into numpy arrays.
  """
  output = tree_map(np.asarray, default_collate(batch))
  return output

def flatten_and_cast(pic):
  """Convert PIL image to flat (1-dimensional) numpy array.
  np.ravel(...) -> Aus einer 28x28 matrix wird ein Vektor mit 784 Elementen"""
  return np.ravel(np.array(pic, dtype=jnp.float32))


def prepareData(dataset_name: str, batchSize: int)-> TrainingData:

    if dataset_name == "mnist":
        DatasetClass = datasets.MNIST
        n_targets = 10 # 10 Klassen (Ziffern 0-9)
    elif dataset_name == "fashionmnist":
        DatasetClass = datasets.FashionMNIST
        n_targets = 10 # 10 Klassen (Ziffern 0-9)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # define our dataset, using torch datasets
    dataset = DatasetClass('_dataset/', download=True, transform=flatten_and_cast)
    # create pytorch data loader with custom collate function
    training_generator = DataLoader(dataset, batch_size=batchSize, collate_fn=numpy_collate)

    # Train dataset
    train_images = np.asarray(dataset.data, dtype=jnp.float32).reshape(len(dataset.data), -1)
    train_labels = one_hot(np.asarray(dataset.targets, dtype=jnp.float32), n_targets)

    # Test dataset
    testDataSet = DatasetClass('_dataset/', download=True, train=False)
    test_images = jnp.asarray(testDataSet.data.numpy().reshape(len(testDataSet.data), -1),
                              dtype=jnp.float32)
    test_labels = one_hot(np.asarray(testDataSet.targets, dtype=jnp.float32), n_targets)

    return  TrainingData(
            training_generator=training_generator,
            train_images=train_images,
            train_labels=train_labels,
            test_images=test_images,
            test_labels=test_labels,
            n_targets=n_targets
        )

def train(data: TrainingData, config: TrainingConfiguration) -> TrainingResults:
    # Reset parameters
    randomKey = config.randomKey
    train_images = data.train_images
    train_labels = data.train_labels
    test_images = data.test_images
    test_labels = data.test_labels
    n_targets = data.n_targets
    layer_sizes = config.layerSizes

    params = init_network_params(layer_sizes, randomKey)

    # Tracking-Listen
    train_accs = []
    test_accs = []
    train_losses = []
    test_losses = []
    epoch_times = []

    print("Training gestartet...")
    print("-" * 60)
    print(" Schritt für Schritt ")

    for epoch in range(config.numEpochs):
        start_time = time.time()
        for x, y in data.training_generator:
            y = one_hot(y, data.n_targets)
            params = update(params, x, y, config.learningRate, grad_calculator_backward)

        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)

        # Metriken berechnen
        train_acc = accuracy(params, train_images, train_labels)
        test_acc = accuracy(params, test_images, test_labels)
        train_loss_val = loss_fn(params, train_images, train_labels)
        test_loss_val = loss_fn(params, test_images, test_labels)

        # Tracking
        train_accs.append(float(train_acc))
        test_accs.append(float(test_acc))
        train_losses.append(float(train_loss_val))
        test_losses.append(float(test_loss_val))

        # Status ausgeben
        print(f"Epoch {epoch + 1:2d}/{config.numEpochs} | "
              f"Zeit: {epoch_time:5.2f}s | "
              f"Train Acc: {train_acc:.4f} | "
              f"Test Acc: {test_acc:.4f} | "
              f"Train Loss: {train_loss_val:.4f} | "
              f"Test Loss: {test_loss_val:.4f}")


    print("-" * 60)
    print(f"Training abgeschlossen! Durchschnittliche Zeit pro Epoch: {np.mean(epoch_times):.2f}s")

    return TrainingResults( train_accs=train_accs,
                            test_accs=test_accs,
                            train_loss=train_losses,
                            test_loss=test_losses,
                            final_params=params,
                            epoch_times=epoch_times
                            )



def createTrainingConfiguration(loadedConfig: dict) -> TrainingConfiguration:
    numEpochs = int(loadedConfig["numEpochs"])
    learningRate = float(loadedConfig["learningRate"])
    batchSize = int(loadedConfig["batchSize"])
    randomKey = random.PRNGKey(int(loadedConfig.get("randomSeed", 0)))
    layer_sizes = list(loadedConfig["layerSizes"])

    buildAutoBatchedFktForJAX()

    return TrainingConfiguration(learningRate=learningRate, numEpochs=numEpochs, batchSize=batchSize, randomKey=randomKey, layerSizes=layer_sizes)

def checkConfigIfMultipleDatasets(config: dict) -> bool:
    return len(list(configFile["dataset"])) > 1

def loadConfigFile()-> dict:
    with open("pureJAX/Configuration/config.yaml", "r") as file:
        config = yaml.safe_load(file)
    return config

if __name__ == "__main__":
    # Training mit Visualisierung durchführen
    configFile = loadConfigFile()
    trainingConfiguration = createTrainingConfiguration(configFile)

    if checkConfigIfMultipleDatasets(configFile):
        # Multiple datasets - train on each one sequentially
        for dataset_name in configFile["dataset"]:
            print(f"\nTraining on dataset: {dataset_name}")
            TrainData = prepareData(dataset_name, trainingConfiguration.batchSize)
            TrainingResult = train(TrainData, trainingConfiguration)

            # Modify config for this specific dataset
            dataset_config = configFile.copy()
            dataset_config["dataset"] = dataset_name

            saving.save_training_session(TrainingResult, dataset_config)
    else:
        # Single dataset trainings
        dataset_name = configFile["dataset"][0]
        TrainData = prepareData(dataset_name, trainingConfiguration.batchSize)
        TrainingResult = train(TrainData, trainingConfiguration)
        saving.save_training_session(TrainingResult, configFile)
