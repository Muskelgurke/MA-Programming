import numpy as np
from typing import List, Tuple, Literal
import matplotlib.pyplot as plt
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
# sizes = numbeer of neurons in each layer

def init_network_params(sizes: List[int], key: Array) -> List[Tuple[Array, Array]]:
  keys = random.split(key, len(sizes))
  return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

layer_sizes = [784, 512, 512, 10]


n_targets = 10 # handschriftliche Ziffern 0-9
params = init_network_params(layer_sizes, random.key(0))

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

"""
# This works on single examples
random_flattened_image = random.normal(random.key(1), (28 * 28,))
preds = predict(params, random_flattened_image)
print(preds.shape)
# Doesn't work with a batch
random_flattened_images = random.normal(random.key(1), (10, 28 * 28))
try:
  preds = predict(params, random_flattened_images)
except TypeError:
  print('Invalid shapes!')
"""

# --- Auto-Batched version of predict ---

# Make a batched version of the `predict` function
batched_predict = vmap(forward, in_axes=(None, 0))
random_flattened_images = random.normal(random.key(1), (10, 28*28))

# `batched_predict` has the same call signature as `predict`
batched_preds = batched_predict(params, random_flattened_images)

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
    global training_generator, train_images, train_labels, test_images, test_labels

    if dataset_name.lower() == "mnist":
        DatasetClass = datasets.MNIST
        n_targets = 10
    elif dataset_name.lower() == "fashionmnist":
        DatasetClass = datasets.FashionMNIST
        n_targets = 10
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

def train_with_visualization(num_epochs: int) -> Tuple[List[float], List[float], List[float], List[float], List[Tuple[Array, Array]]]:
    # Reset parameters
    params = init_network_params(layer_sizes, random.key(0))

    # Tracking-Listen
    log_acc_train = []
    log_acc_test = []
    train_loss = []
    test_losses = []
    epoch_times = []

    print("Training gestartet...")
    print("-" * 60)

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

    return log_acc_train, log_acc_test, train_loss, test_losses, params


def plot_accuracy_curve(train_accuracies: List[float], test_accuracies: List[float],
                        title: str = "Model Accuracy over Epochs"):
    """
    Plottet die Accuracy-Kurven für Training und Test-Daten.

    Args:
        train_accuracies: Liste der Training-Accuracies pro Epoch
        test_accuracies: Liste der Test-Accuracies pro Epoch
        title: Titel des Plots
    """
    epochs = range(1, len(train_accuracies) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accuracies, 'b-o', label='Training Accuracy', linewidth=2, markersize=6)
    plt.plot(epochs, test_accuracies, 'r-s', label='Test Accuracy', linewidth=2, markersize=6)

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(0.5, len(train_accuracies) + 0.5)
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.show()
    pass


def plot_comprehensive_metrics(train_accuracies: List[float], test_accuracies: List[float],
                              train_losses: List[float] = None, test_losses: List[float] = None):
    """
    Plottet umfassende Metriken.

    Args:
        train_accuracies: Training-Accuracies
        test_accuracies: Test-Accuracies
        train_losses: Training-Losses (optional)
        test_losses: Test-Losses (optional)
    """
    epochs = range(1, len(train_accuracies) + 1)

    # Bestimme Anzahl der Subplots
    n_plots = 2 if (train_losses is not None and test_losses is not None) else 1

    fig, axes = plt.subplots(1, n_plots, figsize=(15, 6))
    if n_plots == 1:
        axes = [axes]

    # Accuracy Plot
    ax1 = axes[0]
    ax1.plot(epochs, train_accuracies, 'b-o', label='Training Accuracy', linewidth=2, markersize=6)
    ax1.plot(epochs, test_accuracies, 'r-s', label='Test Accuracy', linewidth=2, markersize=6)
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)

    # Loss Plot (falls verfügbar)
    if n_plots == 2 and train_losses is not None and test_losses is not None:
        ax2 = axes[1]
        ax2.plot(epochs, train_losses, 'b-o', label='Training Loss', linewidth=2, markersize=6)
        ax2.plot(epochs, test_losses, 'r-s', label='Test Loss', linewidth=2, markersize=6)
        ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Training mit Visualisierung durchführen
    with open("Configuration/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    learningRate = config["learningRate"]
    numEpochs = config["numEpochs"]
    batchSize = config["batchSize"]


    prepareData("mnist")

    train_accs, test_accs, train_losses, test_losses, final_params = train_with_visualization(numEpochs)
    # Visualisierungen erstellen
    plot_accuracy_curve(train_accs, test_accs, "MNIST Classification Accuracy")
    plot_comprehensive_metrics(train_accs, test_accs, train_losses, test_losses)