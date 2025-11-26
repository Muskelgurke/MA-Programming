
def calculate_num_bits_with_config(config, num_values) -> int:
    return config.num_bits * num_values


def calculate_for_linear(l: Linear) -> tuple[int, int, int, int, int, int]:
    inputs = int(l.inputs)
    outputs = int(l.outputs)
    out_config = l.output_config
    weight_config = l.weight_config
    bias_config = l.bias_config

    num_weights = inputs * outputs
    if l.bias is not None:
        num_bias = outputs
    else:
        num_bias = 0

    weights_size = calculate_num_bits_with_config(weight_config, num_weights)
    biases_size = calculate_num_bits_with_config(bias_config, num_bias)
    size_params = weights_size + biases_size
    size_outputs = calculate_num_bits_with_config(out_config, outputs)

    multiply_operations = num_weights

    if out_config.stochastic_rounding:
        stochastic_rounding_operations = outputs
    else:
        stochastic_rounding_operations = 0

    add_operations = multiply_operations + biases_size + stochastic_rounding_operations

    rounding_operations = outputs

    return size_params, size_outputs, multiply_operations, add_operations, rounding_operations, stochastic_rounding_operations


def calculate_for_conv1d(outputs_prev_layer: int, l: Conv1d) -> tuple[int, int, int, int, int, int]:
    inputs = outputs_prev_layer
    kernel_size = int(l.kernel_size[0])
    in_channels = int(l.in_channels)
    out_channels = int(l.out_channels)

    if l.padding == "same":
        outputs = inputs
    else:
        outputs = (inputs - kernel_size + 1)

    num_weights = in_channels * out_channels * kernel_size

    if l.bias is not None:
        num_bias = out_channels
    else:
        num_bias = 0

    weights_size = calculate_num_bits_with_config(l.weight_config, num_weights)
    biases_size = calculate_num_bits_with_config(l.bias_config, num_bias)
    size_params = weights_size + biases_size
    size_outputs = calculate_num_bits_with_config(l.output_config, outputs)

    if l.stochastic_rounding:
        stochastic_rounding_operations = outputs * out_channels
    else:
        stochastic_rounding_operations = 0

    multiply_operations = inputs * in_channels * out_channels * kernel_size
    add_operations = multiply_operations + num_bias + stochastic_rounding_operations

    rounding_operations = outputs * out_channels

    return size_params, size_outputs, multiply_operations, add_operations, rounding_operations, stochastic_rounding_operations

