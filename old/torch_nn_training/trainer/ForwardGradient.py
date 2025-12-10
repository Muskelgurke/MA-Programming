
import torch
import torch.nn as nn

from typing import Dict, Tuple, List, Optional

class ForwardGradient:
    """Handles forward gradient estimation using JVP."""

    @staticmethod
    def compute_forward_gradient(model: nn.Module,
                                 inputs: torch.Tensor,
                                 targets: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Compute forward gradient using JVP."""
        named_params = dict(model.named_parameters())
        params = tuple(named_params.values())
        param_names = tuple(named_params.keys())

        # Sample perturbation vectors
        v_params = tuple([torch.randn_like(p) for p in params])

        def loss_fn(params_tuple, x, y):
            params_dict = dict(zip(param_names, params_tuple))
            output = torch.func.functional_call(model, params_dict, x)
            return nn.functional.cross_entropy(output, y)

        # Compute JVP (Forward AD)
        loss, directional_derivative = torch.func.jvp(
            lambda p: loss_fn(p, inputs, targets),
            (params,),
            (v_params,)
        )
        # Compute estimated gradients: gradient = v * directional_derivative
        estimated_gradients = [directional_derivative * v for v in v_params]

        return loss, estimated_gradients