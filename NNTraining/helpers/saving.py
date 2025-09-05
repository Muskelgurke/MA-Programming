import datetime
import pickle
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import jax.numpy as jnp
from jax import Array
import toml

from NNTraining.helpers import plotting

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
            "dataset": config["dataset"]
        },
        "model_configuration": config,
        "training_metrics": {
            "train_accuracies": train_accs,
            "test_accuracies": test_accs,
            "train_losses": train_losses,
            "test_losses": test_losses,
            "epoch_times": epoch_times
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

    plotting.plot_performance(train_losses, train_accs, test_accs, epoch_times, training_dir, config["dataset"])

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
            )
        log_file.write(f"Training abgeschlossen! Durchschnittliche Zeit pro Epoch: {np.mean(epoch_times):.2f}s")