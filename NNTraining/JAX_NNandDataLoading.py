import datetime
import pickle
import json

import numpy as np
from typing import List, Tuple, Literal, Dict, Any
import toml

from NNTraining.helpers import plotting
from pathlib import Path
import time
import yaml
from torch.utils.data import DataLoader, default_collate
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

def init_network_params(sizes: List[int], key: Array) -> List[Tuple[Array, Array]]:
  keys = random.split(key, len(sizes))
  return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

def relu(x: Array) -> Array:
  return jnp.maximum(0, x)

def forward(params: List[Tuple[Array, Array]], image: Array) -> Array:
  """ Predict the class of a single"""
  # per-example predictions
  activations = image
  for w, b in params[:-1]:
    outputs = jnp.dot(w, activations) + b # Neuronen berechnung y = w * x + b
    activations = relu(outputs) # Aktivierung meiner Neuronen

  final_w, final_b = params[-1]
  logits = jnp.dot(final_w, activations) + final_b
  return logits - logsumexp(logits) # Bessere und numerisch stabilere Softmax

def one_hot(x: np.ndarray, k: int, dtype=jnp.float32) -> Array:
  """Create a one-hot encoding of x of size k.
  1.Klassen-Indizes -> jnp.arange(k)
  2.x[:,None] -> x in Spaltenvektor umwandeln
  3.Boolschen werte  in den vorgegebenen dtype umwandeln
  """
  return jnp.array(x[:, None] == jnp.arange(k), dtype)

def accuracy(params: List[Tuple[Array, Array]], images: Array, targets:Array)-> Array:
  target_class = jnp.argmax(targets, axis=1)
  predicted_class = jnp.argmax(batched_predict(params,images), axis=1)
  return jnp.mean(predicted_class == target_class)

def loss(params: List[Tuple[Array, Array]], images:Array, targets:Array)-> Array:
  """ Negative Log-Likelihood(NLL) eine Form von Cross-Entropy-Loss"""
  preds = batched_predict(params,images)
  return -jnp.mean(preds*targets)

@jit
def update(params: List[Tuple[Array,Array]], x: Array, y: Array)-> List[Tuple[Array,Array]]:
  grads = grad(loss)(params, x, y)
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


def prepareData(dataset_name: Literal["mnist", "fashionmnist"]):
    global training_generator, train_images, train_labels, test_images, test_labels, n_targets

    if dataset_name.lower() == "mnist":
        DatasetClass = datasets.MNIST
        n_targets = 10 # 10 Klassen (Ziffern 0-9)
    elif dataset_name.lower() == "fashionmnist":
        DatasetClass = datasets.FashionMNIST
        n_targets = 10 # 10 Klassen (Ziffern 0-9)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # define our dataset, using torch datasets
    dataset = DatasetClass('dataset/', download=True, transform=flatten_and_cast)
    # create pytorch data loader with custom collate function
    training_generator = DataLoader(dataset, batch_size=batchSize, collate_fn=numpy_collate)
    # Get the full train dataset (for checking accuracy while training)
    train_images = np.asarray(dataset.data, dtype=jnp.float32).reshape(len(dataset.data), -1)
    train_labels = one_hot(np.asarray(dataset.targets, dtype=jnp.float32), n_targets)
    # Get full test dataset
    testDataSet = DatasetClass('dataset/', download=True, train=False)
    test_images = jnp.asarray(testDataSet.data.numpy().reshape(len(testDataSet.data), -1),
                              dtype=jnp.float32)
    test_labels = one_hot(np.asarray(testDataSet.targets, dtype=jnp.float32), n_targets)

def save_jax_array(array: Array, filepath: Path) -> None:
    """Save JAX array preserving its type information"""
    # Convert to numpy for storage but preserve the original shape and dtype
    np_array = np.array(array)
    with open(filepath, 'wb') as f:
        # Save metadata and array data
        metadata = {
            'shape': np_array.shape,
            'dtype': str(np_array.dtype),
            'is_jax': True
        }
        np.savez(filepath, data=np_array, metadata=metadata)

def load_jax_array(filepath: Path) -> Array:
    """Load JAX array from file"""
    data = np.load(filepath)
    array_data = data['data']
    metadata = data['metadata'].item() if 'metadata' in data else {}

    # Convert back to JAX array
    jax_array = jnp.array(array_data, dtype=getattr(jnp, metadata.get('dtype', 'float32')))
    return jax_array


def save_training_session(train_accs: List[float], test_accs: List[float],
                          train_losses: List[float], test_losses: List[float],
                          final_params: List[Tuple[Array, Array]],epoch_times: List[float],
                          config: Dict[str, Any]):
    """
    Save the training session including model parameters, metrics, and configuration.
ss
    Returns:
        Path to the created training directory
    """
    # Create timestamp-based directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    training_dir = Path(f"training_sessions/{timestamp}")
    training_dir.mkdir(parents=True, exist_ok=True)

    # Prepare training data for TOML
    training_data = {
        "metadata": {
            "timestamp": timestamp,
            "training_duration": f"{np.sum(epoch_times):.2f}s",
            "average_epoch_time": f"{np.mean(epoch_times):.2f}s",
            "final_train_accuracy": float(train_accs[-1]),
            "final_test_accuracy": float(test_accs[-1]),
            "final_train_loss": float(train_losses[-1]),
            "final_test_loss": float(test_losses[-1]),
            "dataset": config.get("dataset", "mnist")
        },
        "model_configuration": config,
        "training_metrics": {
            "train_accuracies": train_accs,
            "test_accuracies": test_accs,
            "train_losses": train_losses,
            "test_losses": test_losses,
            "epoch_times": epoch_times
        },
        "model_architecture": {
            "layer_sizes": layer_sizes,
            "learning_rate": learningRate,
            "batch_size": batchSize,
            "num_epochs": numEpochs,
            "random_seed": config.get("random_seed", 0)
        }
    }


    # Save TOML file
    toml_path = training_dir / "training_info.toml"
    with open(toml_path, 'w') as f:
        toml.dump(training_data, f)

    # Save model parameters using JAX-aware saving
    model_dir = training_dir / "model_params"
    model_dir.mkdir(exist_ok=True)

    for i, (w, b) in enumerate(final_params):
        save_jax_array(w, model_dir / f"layer_{i}_weights.npz")
        save_jax_array(b, model_dir / f"layer_{i}_biases.npz")

    # Alternative: Save entire params as pickle (preserves JAX types exactly)
    with open(training_dir / "model_params.pkl", 'wb') as f:
        pickle.dump(final_params, f)

    plotting.plot_performance(train_losses, train_accs, test_accs, epoch_times, training_dir)

    # Save configuration as JSON for easy reading
    with open(training_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    with open(training_dir / "training_log.txt", 'a') as log_file:
        totalEpoch = len(train_accs)
        for epoch in range(len(train_accs)):
            epoch_time = epoch_times[epoch]
            train_acc = train_accs[epoch]
            test_acc = test_accs[epoch]
            train_loss = train_losses[epoch]
            test_loss = test_losses[epoch]
            log_file.write(
                f"Epoch {epoch + 1:2d}/{totalEpoch} | "
                f"Zeit: {epoch_time:5.2f}s | "
                f"Train Acc: {train_acc:.4f} | "
                f"Test Acc: {test_acc:.4f} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Test Loss: {test_loss:.4f}\n"
                f"Training abgeschlossen! Durchschnittliche Zeit pro Epoch: {np.mean(epoch_times):.2f}s"
            )


def train(num_epochs: int, randomKey: Array) -> Tuple[List[float], List[float], List[float], List[float], List[Tuple[Array, Array]], List[float]]:
    # Reset parameters
    params = init_network_params(layer_sizes, randomKey)

    # Tracking-Listen
    log_acc_train = []
    log_acc_test = []
    train_loss = []
    test_losses = []
    epoch_times = []

    print("Training gestartet...")
    print("-" * 60)
    print(" Schritt für Schritt ")

    for epoch in range(num_epochs):
        start_time = time.time()
        for x, y in training_generator:
            y = one_hot(y, n_targets)
            params = update(params, x, y)

        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)

        # Metriken berechnen
        train_acc = accuracy(params, train_images, train_labels)
        test_acc = accuracy(params, test_images, test_labels)
        train_loss_val = loss(params, train_images, train_labels)
        test_loss_val = loss(params, test_images, test_labels)

        # Tracking
        log_acc_train.append(float(train_acc))
        log_acc_test.append(float(test_acc))
        train_loss.append(float(train_loss_val))
        test_losses.append(float(test_loss_val))

        # Status ausgeben
        print(f"Epoch {epoch + 1:2d}/{num_epochs} | "
              f"Zeit: {epoch_time:5.2f}s | "
              f"Train Acc: {train_acc:.4f} | "
              f"Test Acc: {test_acc:.4f} | "
              f"Train Loss: {train_loss_val:.4f} | "
              f"Test Loss: {test_loss_val:.4f}")


    print("-" * 60)
    print(f"Training abgeschlossen! Durchschnittliche Zeit pro Epoch: {np.mean(epoch_times):.2f}s")

    return log_acc_train, log_acc_test, train_loss, test_losses, params, epoch_times

def buildAutoBatchedFktForJAX():
    global batched_predict
    batched_predict = vmap(forward, in_axes=(None, 0))

def initializingConfigurationOfTraining():
    with open("Configuration/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    numEpochs = int(config["numEpochs"])
    learningRate = float(config["learningRate"])
    batchSize = int(config["batchSize"])
    layer_sizes = list(config["layerSizes"])
    randomKey = random.PRNGKey(int(config.get("randomSeed", 0)))
    params = init_network_params(layer_sizes, random.key(0))
    buildAutoBatchedFktForJAX()

    return learningRate, numEpochs, batchSize, layer_sizes, params, randomKey, config

if __name__ == "__main__":
    # Training mit Visualisierung durchführen
    learningRate, numEpochs, batchSize, layer_sizes, params, randomKey, config = initializingConfigurationOfTraining()

    prepareData("mnist")

    train_accs, test_accs, train_loss, test_loss, final_params, epoch_times = train(numEpochs,randomKey)

    save_training_session(train_accs,
                          test_accs,
                          train_loss,
                          test_loss,
                          final_params,
                          epoch_times,
                          config)

