
import torch.nn as nn


def calculate_linear_parameters(nn_linear: nn.Linear) -> tuple[int, int]:
    in_features = nn_linear.in_features
    out_features = nn_linear.out_features

    num_weights = in_features * out_features
    num_bias = out_features if nn_linear.bias is not None else 0

    return num_weights, num_bias

def calculate_linear_macs(nn_linear: nn.Linear) -> int:
    in_features = nn_linear.in_features
    out_features = nn_linear.out_features

    macs = in_features * out_features

    return macs

def analyse_linear(nn_linear: nn.Linear) -> None:
    num_weights, num_bias = calculate_linear_parameters(nn_linear)
    macs = calculate_linear_macs(nn_linear)

    print(f"Linear Layer Analysis:")
    print(f"Input Features: {nn_linear.in_features}")
    print(f"Output Features: {nn_linear.out_features}")
    print(f"Number of Weights: {num_weights}")
    print(f"Number of Biases: {num_bias}")
    print(f"MACs (Multiply-Accumulate Operations): {macs}")


def calculate_conv2d_parameters(nn_conv: nn.Conv2d) -> tuple[int, int]:
    """
    Berechnet die Anzahl der Gewichte und Biases.
    Formel: Out_Channels * (In_Channels / Groups) * Kernel_H * Kernel_W
    """
    out_channels = nn_conv.out_channels
    in_channels = nn_conv.in_channels
    groups = nn_conv.groups
    kernel_size = nn_conv.kernel_size  # Ist immer ein Tuple (kh, kw)

    # Anzahl der Weights: C_out * (C_in / groups) * K_h * K_w
    num_weights = out_channels * (in_channels // groups) * kernel_size[0] * kernel_size[1]

    num_bias = out_channels if nn_conv.bias is not None else 0

    return num_weights, num_bias

def calculate_conv2d_mac(nn_conv: nn.Conv2d, input_height: int, input_width: int) -> int:
    """
    Berechnet die MACs für Conv2d Layer.
    Formel: Out_Channels * Output_Height * Output_Width * (In_Channels / Groups) * Kernel_H * Kernel_W
    """
    out_channels = nn_conv.out_channels
    in_channels = nn_conv.in_channels
    groups = nn_conv.groups
    kernel_size = nn_conv.kernel_size  # Ist immer ein Tuple (kh, kw)

    # Berechne die Output-Höhe und -Breite basierend auf den Conv2d-Parametern
    stride_h, stride_w = nn_conv.stride
    padding_h, padding_w = nn_conv.padding
    dilation_h, dilation_w = nn_conv.dilation

    output_height = (input_height + 2 * padding_h - dilation_h * (kernel_size[0] - 1) - 1) // stride_h + 1
    output_width = (input_width + 2 * padding_w - dilation_w * (kernel_size[1] - 1) - 1) // stride_w + 1

    # Berechne die MACs
    macs = out_channels * output_height * output_width * (in_channels // groups) * kernel_size[0] * kernel_size[1]

    return macs

def calculate_conv1d_mac(nn_conv: nn.Conv1d, input_length: int) -> int:
    """
    Berechnet die MACs für Conv1d Layer.
    Formel: Out_Channels * Output_Length * (In_Channels / Groups) * Kernel_Size
    """
    out_channels = nn_conv.out_channels
    in_channels = nn_conv.in_channels
    groups = nn_conv.groups
    kernel_size = nn_conv.kernel_size[0]

    # Berechne die Output-Länge basierend auf den Conv1d-Parametern
    stride = nn_conv.stride[0]
    padding = nn_conv.padding[0]
    dilation = nn_conv.dilation[0]

    output_length = (input_length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

    # Berechne die MACs
    macs = out_channels * output_length * (in_channels // groups) * kernel_size

    return macs
def calculate_conv1d_parameters(nn_conv: nn.Conv1d) -> tuple[int, int]:
    """
    Berechnet die Anzahl der Gewichte und Biases für Conv1d.
    Formel: Out_Channels * (In_Channels / Groups) * Kernel_Size
    """
    out_channels = nn_conv.out_channels
    in_channels = nn_conv.in_channels
    groups = nn_conv.groups

    # Kernel Size ist bei Conv1d ein Tuple mit einem Element, z.B. (3,)
    kernel_size = nn_conv.kernel_size[0]

    # Anzahl der Weights
    num_weights = out_channels * (in_channels // groups) * kernel_size

    num_bias = out_channels if nn_conv.bias is not None else 0

    return num_weights, num_bias

def calculate_sequential_parameters(nn_seq: nn.Sequential) -> tuple[int, int]:
    total_weights = 0
    total_biases = 0

    for layer in nn_seq:
        if isinstance(layer, nn.Linear):
            weights, biases = calculate_linear_parameters(layer)
            total_weights += weights
            total_biases += biases
        elif isinstance(layer, nn.Conv2d):
            weights, biases = calculate_conv2d_parameters(layer)
            total_weights += weights
            total_biases += biases
        elif isinstance(layer, nn.Conv1d):
            weights, biases = calculate_conv1d_parameters(layer)
            total_weights += weights
            total_biases += biases
        # Weitere Layer-Typen können hier hinzugefügt werden

    return total_weights, total_biases

def calculate_sequential_macs(nn_seq: nn.Sequential, input_shape: tuple) -> int:
    total_macs = 0
    current_shape = input_shape  # Start mit der Eingabeform

    for layer in nn_seq:
        if isinstance(layer, nn.Linear):
            macs = calculate_linear_macs(layer)
            total_macs += macs
            current_shape = (current_shape[0], layer.out_features)  # Update Form
        if isinstance(layer, nn.Conv1d):
            macs = calculate_conv1d_mac(layer, current_shape[2])
            total_macs += macs
            # Update Form: (Batch, Channels, Length)
            stride = layer.stride[0]
            padding = layer.padding[0]
            dilation = layer.dilation[0]
            kernel_size = layer.kernel_size[0]
            output_length = (current_shape[2] + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
            current_shape = (current_shape[0], layer.out_channels, output_length)
        elif isinstance(layer, nn.Conv2d):
            macs = calculate_conv2d_mac(layer, current_shape[2], current_shape[3])
            total_macs += macs
            # Update Form: (Batch, Channels, Height, Width)
            stride_h, stride_w = layer.stride
            padding_h, padding_w = layer.padding
            dilation_h, dilation_w = layer.dilation
            kernel_size_h, kernel_size_w = layer.kernel_size
            output_height = (current_shape[2] + 2 * padding_h - dilation_h * (kernel_size_h - 1) - 1) // stride_h + 1
            output_width = (current_shape[3] + 2 * padding_w - dilation_w * (kernel_size_w - 1) - 1) // stride_w + 1
            current_shape = (current_shape[0], layer.out_channels, output_height, output_width)

    return total_macs

linear_Test = nn.Linear(10,10)
conv_test = nn.Conv2d(3,16, kernel_size=3, padding=1)

model = nn.Sequential(linear_Test, linear_Test)

print(f"{calculate_sequential_macs(model , (1,10,32,32))}")