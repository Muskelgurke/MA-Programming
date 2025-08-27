import jax.numpy as jnp
from typing import List, Tuple, Any, Callable
from jax import Array
from jax.random import PRNGKey
import numpy as np
from jax.tree_util import tree_map
from torch.utils.data import DataLoader, default_collate
from torchvision.datasets import MNIST
import time

from jax import grad, jit, vmap
from jax import random
from jax.scipy.special import logsumexp
from triton.language import dtype


# We need a fucntion to intilize the weights and biases for a dense neural network layer

def random_layer_params(m: int,n: int,key: int,scale: float=1e-2) -> tuple[Array, Array]:
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n,m)), scale * random.normal(b_key, (n,))

# initialize all layers for a fully-connected NN with sizes
# sizes = numbeer of neurons in each layer
# keys =
def init_network_params(sizes: List[int], key: Array) -> List[Tuple[Array, Array]]:
  keys = random.split(key, len(sizes))
  return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

layer_sizes = [784, 512, 512, 10]
step_size = 0.01
num_epochs = 8
batch_size = 128
n_targets = 10 # handschriftliche Ziffern 0-9
params = init_network_params(layer_sizes, random.key(0))

def relu(x: Array) -> Array:
  return jnp.maximum(0, x)

def predict(params: List[Tuple[Array, Array]], image: Array) -> Array:
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
batched_predict = vmap(predict, in_axes=(None, 0))
random_flattened_images = random.normal(random.key(1), (10, 28*28))

# `batched_predict` has the same call signature as `predict`
batched_preds = batched_predict(params, random_flattened_images)
print(batched_preds.shape)

def one_hot(x: np.ndarray, k: int, dtype=jnp.float32) -> Array:
  """Create a one-hot encoding of x of size k.
  1.Klassen-Indizes -> jnp.arange(k)
  2.x[:,None] -> x in Spaltenvektor umwandeln
  3.Boolschen werte  in den vorgegebenen dtype umwandeln
  """
  return jnp.array(x[:, None] == jnp.arange(k), dtype)

def accuracy(params: List[Tuple[Array, Array]], images:Array, targets:Array)-> Array:
  target_class = jnp.argmax(targets, axis=1)
  predicted_class = jnp.argmax(batched_predict(params,images), axis=1)
  return jnp.mean(predicted_class == target_class)

def loss(params: List[Tuple[Array, Array]], images:Array, targets:Array)-> Array:
  """ Negative Log-Likelihood(NLL) eine Form von Cross-Entropy-Loss"""
  preds = batched_predict(params,images)
  return -jnp.mean(preds*targets)

## Mean Squared Error Loss
## Bei Klassifikationsproblemen, kommt es zu langsameren Lernen
def mse_loss(params: List[Tuple[Array, Array]], images:Array, targets:Array)-> Array:
  preds = batched_predict(params,images)
  return jnp.mean((preds - targets)**2)

@jit
def update(params: List[Tuple[Array,Array]], x: Array, y: Array)-> List[Tuple[Array,Array]]:
  grads = grad(loss)(params, x, y)
  return [(w-step_size * dw, b-step_size * db)
          for (w,b), (dw,db)in zip(params, grads)]

def numpy_collate(batch):
  """  Collate function specifies how to combine a list of data samples into a batch.
  default_collate creates pytorch tensors, then tree_map converts them into numpy arrays.
  """
  return tree_map(np.asarray, default_collate(batch))

def flatten_and_cast(pic):
  """Convert PIL image to flat (1-dimensional) numpy array.
  np.ravel(...) -> Aus einer 28x28 matrix wird ein Vektor mit 784 Elementen"""
  return np.ravel(np.array(pic, dtype=jnp.float32))

#define our dataset, using torch datasets
mnist_dataset = MNIST('/tmp/mnist/', download=True, transform=flatten_and_cast)

# create pytorch data loader with custom collate function
training_generator = DataLoader(mnist_dataset, batch_size=batch_size, collate_fn=numpy_collate)

# Get the full train dataset (for checking accuracy while training)
train_images = np.asarray(mnist_dataset.data,dtype=jnp.float32).reshape(len(mnist_dataset.data), -1)
train_labels = one_hot(np.asarray(mnist_dataset.targets,dtype=jnp.float32), n_targets)
# Get full test dataset
mnist_dataset_test = MNIST('/tmp/mnist/', download=True, train=False)
test_images = jnp.asarray(mnist_dataset_test.data.numpy().reshape(len(mnist_dataset_test.data), -1), dtype=jnp.float32)
test_labels = one_hot(np.asarray(mnist_dataset_test.targets,dtype=jnp.float32), n_targets)

for epoch in range(num_epochs):
  start_time = time.time()
  for x, y in training_generator:
    y = one_hot(y, n_targets)
    params = update(params, x, y)
  epoch_time = time.time() - start_time

  train_acc = accuracy(params, train_images, train_labels)
  test_acc = accuracy(params, test_images, test_labels)
  print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
  print("Training set accuracy {}".format(train_acc))
  print("Test set accuracy {}".format(test_acc))