from typing import List, Tuple, Optional
from jax import Array
import jax.numpy as jnp
from jax import random, vmap
from torchgen.model import dispatch_keys


class JAXModel:
    """Klasse zur Kapselung des JAX-Modells"""

    def __init__(self, layer_sizes: List[int], random_key: int):
        self.layer_sizes = layer_sizes
        self.params = self.init_network_params(layer_sizes, random_key)
        self.batched_predict = None

    # initialize all layers for a fully-connected NN with sizes
    # sizes = number of neurons in each layer
    def init_network_params(self, layer_sizes: List[int], random_key: int) -> List[Tuple[Array, Array]]:
        """intitialize all lyers for a fully-connected NN with sizes"""
        layer_keys = random.split(random_key, len(layer_sizes))
        network_parameters = [
            self.initialize_random_layer_params(input_size, output_size, layer_key)
            for input_size, output_size, layer_key in zip(layer_sizes[:-1], layer_sizes[1:], layer_keys)
        ]
        return network_parameters

    # We need a fuction to intilize the weights and biases for a dense neural network layer
    def initialize_random_layer_params(
        self,
        input_dimension: int,
        output_dimension: int,
        random_seed_key: int,
        initialization_scale: float = 1e-2
    ) -> Tuple[Array, Array]:
        weight_key, bias_key = random.split(random_seed_key)
        random_weight_matrix = initialization_scale * random.normal(key=weight_key, shape=(output_dimension, input_dimension))
        random_bias_matrix = initialization_scale * random.normal(bias_key, (output_dimension,))
        return random_weight_matrix, random_bias_matrix