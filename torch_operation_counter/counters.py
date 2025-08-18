from math import prod
from numbers import Number
from typing import Any, List

from torch import ops

from torch_operation_counter.utils import conv_ops_count, transpose_shape


def basic_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Handle cases where inputs[0] might not be a tensor
    if hasattr(inputs[0], 'shape'):
        num_operations = prod(inputs[0].shape)
    else:
        # If inputs[0] is not a tensor (e.g., a scalar), count as 1 operation
        num_operations = 1
    return num_operations


def arange_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    num_operations = len(outputs[0])
    return num_operations


def scatter_add_ops(inputs: List[Any], outputs: List[Any]) -> int:
    index = inputs[2]
    src = inputs[3]
    num_operations = src.numel() + index.numel()
    return num_operations


def scatter_reduce_ops(inputs: List[Any], outputs: List[Any]) -> int:
    reduce = inputs[4].lower()

    if reduce in ["sum", "prod", "amax", "amin"]:
        num_operations = scatter_add_ops(inputs, outputs)
    elif reduce == "mean":
        num_operations = 2 * scatter_add_ops(inputs, outputs)
    else:
        raise ValueError(f"Unknown reduce operation: {reduce}")
    return num_operations


def matmul_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    input_shapes = [v.shape for v in inputs]
    # For matrix multiplication: A @ B, where A is (m, n) and B is (n, p)
    # Each output element requires n multiply-accumulate operations
    # Total FLOPs = m * n * p
    if len(input_shapes[0]) == 2 and len(input_shapes[1]) == 2:
        m, n = input_shapes[0]
        n2, p = input_shapes[1]
        if n != n2:  # Check if matrices are compatible
            raise ValueError(f"Incompatible matrix shapes: {input_shapes[0]} and {input_shapes[1]}")
        return m * n * p
    else:
        # For higher dimensional tensors, use the last two dimensions
        # This handles batched matrix multiplication
        last_dim_a = input_shapes[0][-1]
        last_dim_b = input_shapes[1][-2]
        if last_dim_a != last_dim_b:
            raise ValueError(f"Incompatible tensor shapes for matmul: {input_shapes[0]} and {input_shapes[1]}")
        return prod(input_shapes[0]) * input_shapes[1][-1]


def addmm_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    input_shapes = [v.shape for v in inputs[1:3]]
    batch_size, input_dim = input_shapes[0]
    output_dim = input_shapes[1][1]
    # Matrix multiplication: batch_size * input_dim * output_dim
    # Plus bias addition: batch_size * output_dim
    num_operations = batch_size * input_dim * output_dim + batch_size * output_dim
    return num_operations


def bmm_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    input_shapes = [v.shape for v in inputs]
    # For batched matrix multiplication: (batch, m, n) @ (batch, n, p)
    # Each batch element requires m * n * p operations
    batch_size = input_shapes[0][0]
    m, n = input_shapes[0][1], input_shapes[0][2]
    n2, p = input_shapes[1][1], input_shapes[1][2]
    if n != n2:
        raise ValueError(f"Incompatible matrix shapes for bmm: {input_shapes[0]} and {input_shapes[1]}")
    num_operations = batch_size * m * n * p
    return num_operations


def relu_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    num_operations = 2 * prod(inputs[0].shape)  # also count the comparison
    return num_operations


def leaky_relu_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    num_operations = 4 * prod(inputs[0].shape)  # also count the comparison
    return num_operations


def softmax_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    num_operations = 5 * prod(inputs[0].shape)
    return num_operations


def log_softmax_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    num_operations = 6 * prod(inputs[0].shape)
    return num_operations


def mean_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # For mean reduction: sum all elements + divide by count
    # If dims is specified, reduce only those dimensions
    # If no dims specified, reduce all dimensions
    if len(inputs) > 1 and inputs[1] is not None:
        dims = inputs[1]
        # Sum operations: prod([inputs[0].size(dim) for dim in dims]) - 1 additions
        sum_ops = prod([inputs[0].size(dim) for dim in dims]) - 1
    else:
        # Reduce all dimensions
        sum_ops = prod(inputs[0].shape) - 1
    div_ops = 1
    num_operations = sum_ops + div_ops
    return num_operations


def convolution_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    x, w = inputs[:2]
    x_shape, w_shape, out_shape = (x.shape, w.shape, outputs[0].shape)
    transposed = inputs[6]
    return conv_ops_count(x_shape, w_shape, out_shape, transposed=transposed)


def native_batch_norm_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    num_operations = 2 * prod(inputs[0].shape)
    # count the affine `running_mean` and `running_var` operation
    if inputs[1] is not None and inputs[2] is not None:
        num_operations = 2 * num_operations
    return num_operations


def convolution_backward_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    grad_out_shape, x_shape, w_shape = [i.shape for i in inputs[:3]]
    output_mask = inputs[-1]
    fwd_transposed = inputs[7]
    ops_count = 0
    if output_mask[0]:
        grad_input_shape = outputs[0].shape
        ops_count += conv_ops_count(grad_out_shape, w_shape, grad_input_shape, not fwd_transposed)
    if output_mask[1]:
        grad_weight_shape = outputs[1].shape
        ops_count += conv_ops_count(transpose_shape(x_shape), grad_out_shape, grad_weight_shape, fwd_transposed)
    return ops_count


def max_pool2d_with_indices_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    input_tensor = inputs[0]
    kernel_size = inputs[1] if len(inputs) > 1 else (2, 2)
    stride = inputs[2] if len(inputs) > 2 else (2, 2)
    padding = inputs[3] if len(inputs) > 3 else (0, 0)
    # Calculate output dimensions
    nC = input_tensor.shape[1]
    nH = (input_tensor.shape[2] - kernel_size[0] + 2 * padding[0]) // stride[0] + 1
    nW = (input_tensor.shape[3] - kernel_size[1] + 2 * padding[1]) // stride[1] + 1
    # Number of comparisons per window
    comparisons_per_window = prod(kernel_size) - 1
    # Total operations are comparisons across all output elements
    num_operations = comparisons_per_window * nC * nH * nW
    return num_operations


def avg_pool2d_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    input_tensor = inputs[0]
    kernel_size = inputs[1] if len(inputs) > 1 else (2, 2)
    stride = inputs[2] if len(inputs) > 2 else (2, 2)
    padding = inputs[3] if len(inputs) > 3 else (0, 0)
    # Calculate output dimensions
    nC = input_tensor.shape[1]
    nH = (input_tensor.shape[2] - kernel_size[0] + 2 * padding[0]) // stride[0] + 1
    nW = (input_tensor.shape[3] - kernel_size[1] + 2 * padding[1]) // stride[1] + 1
    # For average pooling: sum all elements in window, then divide by window size
    # Each output element requires (kernel_size - 1) additions + 1 division
    operations_per_window = prod(kernel_size) - 1 + 1  # sum + divide
    num_operations = operations_per_window * nC * nH * nW
    return num_operations


def constant_pad_nd_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Padding operations: each padded element requires a write operation
    # The number of padded elements is the difference between output and input sizes
    input_size = prod(inputs[0].shape)
    output_size = prod(outputs[0].shape)
    num_operations = output_size - input_size
    return num_operations


def max_pool2d_with_indices_backward_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    grad_tensor = inputs[0]  # Assuming this is the gradient tensor
    # Each element in the gradient tensor corresponds to an operation
    num_operations = prod(grad_tensor.shape)
    return num_operations


def native_batch_norm_backward_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    num_operations = 2 * prod(inputs[0].shape)
    return num_operations


def threshold_backward_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    num_operations = 2 * prod(inputs[0].shape)
    return num_operations


def gather_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    index_tensor = inputs[2]  # Assuming this is the index tensor
    num_operations = prod(index_tensor.shape)
    return num_operations


def index_add_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    source_tensor = inputs[3]  # Assuming this is the source tensor
    num_operations = 2 * prod(source_tensor.shape)
    return num_operations


def new_zeros_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    output_tensor = outputs[0]
    # Each element involves a write operation
    num_operations = prod(output_tensor.shape)
    return num_operations


def masked_fill_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    mask_tensor = inputs[1]
    # Count true elements in the mask as operations (read mask, write value)
    # Each true element requires: read mask + read value + write operation = 3 ops
    true_elements = (mask_tensor == True).sum().item()
    num_operations = 3 * true_elements
    return num_operations


def full_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    output_tensor = outputs[0]
    num_operations = prod(output_tensor.shape)  # Each element set once
    return num_operations

def choleksy_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # count of operations is 1/6 * n^3 multiplications and 1/6*n^3 additions.
    # In accordance with
    # - https://web.archive.org/web/20240119062842/https://www.cs.princeton.edu/courses/archive/fall20/cos302/notes/cos302_f20_precept5_lu_cholesky.pdf slide 20
    # - https://web.archive.org/web/20240513153538/https://www.cs.utexas.edu/~flame/Notes/NotesOnCholReal.pdf page 8
    n = inputs[0].shape[-1]
    batch_count = prod(inputs[0].shape[:-2])
    return (1/3 * n**3) * batch_count

def cholesky_inverse_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # The input is L, which is of the same size as the original matrix.
    # count of multiplications is 1/3*n^3 according to https://arxiv.org/pdf/1111.4144 page 3.
    # the count of additions is roughly same: 1/3*n^3.
    n = inputs[0].shape[-1]
    batch_count = prod(inputs[0].shape[:-2])
    return (2/3 * n**3) * batch_count

def layer_norm_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Layer normalization: compute mean, variance, normalize, scale and shift
    # For each element: 2 ops (mean, var) + 2 ops (normalize) + 2 ops (scale, shift)
    num_operations = 6 * prod(inputs[0].shape)
    return num_operations


def dropout_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Dropout: generate random mask + apply mask + scale
    # Each element: random generation + comparison + multiplication
    num_operations = 3 * prod(inputs[0].shape)
    return num_operations


def embedding_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Embedding lookup: each index requires a lookup operation
    indices = inputs[1]
    embedding_dim = inputs[0].shape[-1]
    num_operations = prod(indices.shape) * embedding_dim
    return num_operations


def lstm_cell_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # LSTM cell: input, hidden, cell state transformations
    # Each gate (input, forget, cell, output) requires matrix operations
    input_size = inputs[0].shape[-1]
    hidden_size = inputs[1].shape[-1]
    batch_size = inputs[0].shape[0]
    # 4 gates * (input_size + hidden_size) * hidden_size operations per batch
    num_operations = 4 * (input_size + hidden_size) * hidden_size * batch_size
    return num_operations


def cat_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Concatenation: copy elements from input tensors to output
    # Each element requires a copy operation
    num_operations = prod(outputs[0].shape)
    return num_operations


def split_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Split: copy elements from input to multiple outputs
    # Each element requires a copy operation
    total_elements = sum(prod(output.shape) for output in outputs)
    num_operations = total_elements
    return num_operations


def view_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # View/reshape: no actual data movement, just metadata change
    # But we count the shape computation
    num_operations = len(outputs[0].shape)
    return num_operations


def transpose_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Transpose: reorder dimensions, may require data movement
    # Count as the number of elements that need to be moved
    num_operations = prod(outputs[0].shape)
    return num_operations


def sum_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Sum reduction: add all elements along specified dimensions
    # Each element requires an addition operation
    if len(inputs) > 1 and inputs[1] is not None:
        # Sum along specific dimensions
        dims = inputs[1]
        total_elements = prod([inputs[0].size(dim) for dim in dims])
        num_operations = total_elements - 1  # n-1 additions for n elements
    else:
        # Sum all elements
        num_operations = prod(inputs[0].shape) - 1
    return num_operations


def max_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Max reduction: find maximum along specified dimensions
    # Each comparison requires one operation
    if len(inputs) > 1 and inputs[1] is not None:
        # Max along specific dimensions
        dims = inputs[1]
        total_elements = prod([inputs[0].size(dim) for dim in dims])
        num_operations = total_elements - 1  # n-1 comparisons for n elements
    else:
        # Max of all elements
        num_operations = prod(inputs[0].shape) - 1
    return num_operations


def linear_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Linear layer: input @ weight.T + bias
    # Matrix multiplication + bias addition
    input_shape = inputs[0].shape
    weight_shape = inputs[1].shape
    if len(inputs) > 2 and inputs[2] is not None:
        # With bias
        bias_shape = inputs[2].shape
        # Matrix multiplication: batch_size * input_features * output_features
        # Bias addition: batch_size * output_features
        batch_size = input_shape[0] if len(input_shape) > 1 else 1
        input_features = input_shape[-1]
        output_features = weight_shape[0]
        matmul_ops = batch_size * input_features * output_features
        bias_ops = batch_size * output_features
        num_operations = matmul_ops + bias_ops
    else:
        # Without bias
        batch_size = input_shape[0] if len(input_shape) > 1 else 1
        input_features = input_shape[-1]
        output_features = weight_shape[0]
        num_operations = batch_size * input_features * output_features
    return num_operations


def gelu_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # GELU activation: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    # Each element requires: power, multiplication, sqrt, tanh, addition, multiplication
    num_operations = 7 * prod(inputs[0].shape)
    return num_operations


def swish_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Swish activation: x * sigmoid(x)
    # Each element requires: sigmoid + multiplication
    num_operations = 3 * prod(inputs[0].shape)  # sigmoid (3 ops) + multiplication (1 op)
    return num_operations


def unsqueeze_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Unsqueeze: add dimension, no data movement
    # Just metadata change
    num_operations = 1
    return num_operations


def squeeze_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Squeeze: remove dimension, no data movement
    # Just metadata change
    num_operations = 1
    return num_operations


def expand_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Expand: broadcast dimensions, no data movement
    # Just metadata change
    num_operations = 1
    return num_operations


def repeat_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Repeat: copy data multiple times
    # Each element requires a copy operation
    num_operations = prod(outputs[0].shape)
    return num_operations


def stack_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Stack: concatenate tensors along new dimension
    # Each element requires a copy operation
    num_operations = prod(outputs[0].shape)
    return num_operations


def unstack_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Unstack: split tensor along dimension
    # Each element requires a copy operation
    total_elements = sum(prod(output.shape) for output in outputs)
    num_operations = total_elements
    return num_operations


def chunk_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Chunk: split tensor into chunks
    # Each element requires a copy operation
    total_elements = sum(prod(output.shape) for output in outputs)
    num_operations = total_elements
    return num_operations


def narrow_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Narrow: slice tensor, no data movement
    # Just metadata change
    num_operations = 1
    return num_operations


def index_select_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Index select: select elements based on indices
    # Each selected element requires a read operation
    indices = inputs[1]
    if hasattr(indices, 'shape'):
        num_operations = prod(indices.shape)
    else:
        # If indices is a scalar, count as 1 operation
        num_operations = 1
    return num_operations


def masked_select_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Masked select: select elements based on boolean mask
    # Each true element requires a read operation
    mask = inputs[1]
    if hasattr(mask, 'sum'):
        true_elements = (mask == True).sum().item()
    else:
        # If mask is a scalar, count as 1 operation
        true_elements = 1
    num_operations = true_elements
    return num_operations


def take_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Take: select elements based on indices
    # Each selected element requires a read operation
    indices = inputs[1]
    if hasattr(indices, 'shape'):
        num_operations = prod(indices.shape)
    else:
        # If indices is a scalar, count as 1 operation
        num_operations = 1
    return num_operations


def put_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Put: assign values at specified indices
    # Each index requires a write operation
    indices = inputs[1]
    if hasattr(indices, 'shape'):
        num_operations = prod(indices.shape)
    else:
        # If indices is a scalar, count as 1 operation
        num_operations = 1
    return num_operations


def roll_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Roll: shift tensor elements
    # Each element requires a copy operation
    num_operations = prod(outputs[0].shape)
    return num_operations


def flip_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Flip: reverse tensor along specified dimensions
    # Each element requires a copy operation
    num_operations = prod(outputs[0].shape)
    return num_operations


def rot90_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Rot90: rotate tensor by 90 degrees
    # Each element requires a copy operation
    num_operations = prod(outputs[0].shape)
    return num_operations


def diag_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Diag: extract diagonal elements
    # Each diagonal element requires a read operation
    num_operations = prod(outputs[0].shape)
    return num_operations


def tril_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Tril: extract lower triangular matrix
    # Each element requires a read operation
    num_operations = prod(outputs[0].shape)
    return num_operations


def triu_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Triu: extract upper triangular matrix
    # Each element requires a read operation
    num_operations = prod(outputs[0].shape)
    return num_operations


def eye_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Eye: create identity matrix
    # Each diagonal element requires a write operation
    num_operations = prod(outputs[0].shape)
    return num_operations


def ones_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Ones: create tensor filled with ones
    # Each element requires a write operation
    num_operations = prod(outputs[0].shape)
    return num_operations


def zeros_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Zeros: create tensor filled with zeros
    # Each element requires a write operation
    num_operations = prod(outputs[0].shape)
    return num_operations


def randn_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Randn: create tensor with random normal distribution
    # Each element requires a random generation + write operation
    num_operations = 2 * prod(outputs[0].shape)
    return num_operations


def normal_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Normal: create tensor with normal distribution
    # Each element requires a random generation + write operation
    num_operations = 2 * prod(outputs[0].shape)
    return num_operations


def uniform_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Uniform: create tensor with uniform distribution
    # Each element requires a random generation + write operation
    num_operations = 2 * prod(outputs[0].shape)
    return num_operations


def clone_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Clone: copy tensor data
    # Each element requires a copy operation
    num_operations = prod(outputs[0].shape)
    return num_operations


def detach_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Detach: remove gradient computation, no data movement
    # Just metadata change
    num_operations = 1
    return num_operations


def requires_grad_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Requires_grad: set gradient computation flag, no data movement
    # Just metadata change
    num_operations = 1
    return num_operations


def contiguous_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Contiguous: ensure tensor memory layout is contiguous
    # May require data movement if not already contiguous
    if inputs[0].is_contiguous():
        num_operations = 1  # Just metadata check
    else:
        num_operations = prod(outputs[0].shape)  # Data movement
    return num_operations


def to_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # To: convert tensor dtype/device
    # Each element requires a conversion operation
    num_operations = prod(outputs[0].shape)
    return num_operations


def cpu_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # CPU: move tensor to CPU
    # Each element requires a copy operation
    num_operations = prod(outputs[0].shape)
    return num_operations


def cuda_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # CUDA: move tensor to GPU
    # Each element requires a copy operation
    num_operations = prod(outputs[0].shape)
    return num_operations


def half_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Half: convert to half precision
    # Each element requires a conversion operation
    num_operations = prod(outputs[0].shape)
    return num_operations


def float_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Float: convert to float precision
    # Each element requires a conversion operation
    num_operations = prod(outputs[0].shape)
    return num_operations


def long_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Long: convert to long precision
    # Each element requires a conversion operation
    num_operations = prod(outputs[0].shape)
    return num_operations


def int_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Int: convert to int precision
    # Each element requires a conversion operation
    num_operations = prod(outputs[0].shape)
    return num_operations


def bool_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Bool: convert to boolean
    # Each element requires a conversion operation
    num_operations = prod(outputs[0].shape)
    return num_operations


def item_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Item: extract single element from tensor
    # Single read operation
    num_operations = 1
    return num_operations


def numel_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Numel: get number of elements
    # Just metadata access
    num_operations = 1
    return num_operations


def dim_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Dim: get number of dimensions
    # Just metadata access
    num_operations = 1
    return num_operations


def size_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Size: get tensor size
    # Just metadata access
    num_operations = 1
    return num_operations


def is_contiguous_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Is_contiguous: check if tensor memory layout is contiguous
    # Just metadata check
    num_operations = 1
    return num_operations


def is_cuda_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Is_cuda: check if tensor is on CUDA device
    # Just metadata check
    num_operations = 1
    return num_operations


def device_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Device: get tensor device
    # Just metadata access
    num_operations = 1
    return num_operations


def dtype_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Dtype: get tensor data type
    # Just metadata access
    num_operations = 1
    return num_operations


def fill_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Fill_: fill tensor with value
    # Each element requires a write operation
    num_operations = prod(outputs[0].shape)
    return num_operations


def zero_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Zero_: fill tensor with zeros
    # Each element requires a write operation
    num_operations = prod(outputs[0].shape)
    return num_operations


def ones_ops_inplace(inputs: List[Any], outputs: List[Any]) -> Number:
    # Ones_: fill tensor with ones
    # Each element requires a write operation
    num_operations = prod(outputs[0].shape)
    return num_operations


def random_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Random_: fill tensor with random values
    # Each element requires random generation + write operation
    num_operations = 2 * prod(outputs[0].shape)
    return num_operations


def index_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Index: advanced indexing
    # Each selected element requires a read operation
    num_operations = prod(outputs[0].shape)
    return num_operations


def index_copy_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Index_copy_: copy values at specified indices
    # Each index requires a copy operation
    indices = inputs[1]
    num_operations = prod(indices.shape)
    return num_operations


def index_fill_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Index_fill_: fill values at specified indices
    # Each index requires a write operation
    indices = inputs[1]
    num_operations = prod(indices.shape)
    return num_operations


def index_add_ops_advanced(inputs: List[Any], outputs: List[Any]) -> Number:
    # Index_add_: add values at specified indices
    # Each index requires a read + add + write operation
    indices = inputs[1]
    num_operations = 3 * prod(indices.shape)
    return num_operations


def masked_scatter_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Masked_scatter: scatter values based on mask
    # Each true element requires a copy operation
    mask = inputs[1]
    true_elements = (mask == True).sum().item()
    num_operations = true_elements
    return num_operations


def masked_select_ops_advanced(inputs: List[Any], outputs: List[Any]) -> Number:
    # Masked_select: select elements based on boolean mask
    # Each true element requires a read operation
    mask = inputs[1]
    true_elements = (mask == True).sum().item()
    num_operations = true_elements
    return num_operations


def masked_fill_ops_advanced(inputs: List[Any], outputs: List[Any]) -> Number:
    # Masked_fill: fill values based on boolean mask
    # Each true element requires a write operation
    mask = inputs[1]
    true_elements = (mask == True).sum().item()
    num_operations = true_elements
    return num_operations


def masked_scatter_ops_advanced(inputs: List[Any], outputs: List[Any]) -> Number:
    # Masked_scatter: scatter values based on boolean mask
    # Each true element requires a copy operation
    mask = inputs[1]
    true_elements = (mask == True).sum().item()
    num_operations = true_elements
    return num_operations


def gather_ops_advanced(inputs: List[Any], outputs: List[Any]) -> Number:
    # Gather: gather values based on indices
    # Each index requires a read operation
    indices = inputs[2]
    num_operations = prod(indices.shape)
    return num_operations


def scatter_ops_advanced(inputs: List[Any], outputs: List[Any]) -> Number:
    # Scatter: scatter values based on indices
    # Each index requires a write operation
    indices = inputs[2]
    num_operations = prod(indices.shape)
    return num_operations


def scatter_add_ops_advanced(inputs: List[Any], outputs: List[Any]) -> Number:
    # Scatter_add: add values based on indices
    # Each index requires a read + add + write operation
    indices = inputs[2]
    num_operations = 3 * prod(indices.shape)
    return num_operations


def scatter_reduce_ops_advanced(inputs: List[Any], outputs: List[Any]) -> Number:
    # Scatter_reduce: reduce values based on indices
    # Each index requires a read + reduce + write operation
    indices = inputs[2]
    reduce = inputs[4].lower()
    if reduce in ["sum", "prod", "amax", "amin"]:
        num_operations = 3 * prod(indices.shape)
    elif reduce == "mean":
        num_operations = 4 * prod(indices.shape)  # + division
    else:
        num_operations = 3 * prod(indices.shape)
    return num_operations


def topk_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Topk: find top-k values and indices
    # Each element requires a comparison operation
    k = inputs[1] if len(inputs) > 1 else 1
    num_operations = k * prod(inputs[0].shape)
    return num_operations


def sort_ops_advanced(inputs: List[Any], outputs: List[Any]) -> Number:
    # Sort: sort tensor values
    # Each element requires a comparison operation
    # For n elements, approximately n*log(n) comparisons
    n = prod(inputs[0].shape)
    num_operations = int(n * (n ** 0.5))  # Approximation for sorting complexity
    return num_operations


def argsort_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Argsort: sort tensor indices
    # Each element requires a comparison operation
    # For n elements, approximately n*log(n) comparisons
    n = prod(inputs[0].shape)
    num_operations = int(n * (n ** 0.5))  # Approximation for sorting complexity
    return num_operations


def kthvalue_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Kthvalue: find k-th smallest value
    # Each element requires a comparison operation
    k = inputs[1] if len(inputs) > 1 else 1
    num_operations = k * prod(inputs[0].shape)
    return num_operations


def unique_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Unique: find unique elements
    # Each element requires a comparison operation
    # For n elements, approximately n*log(n) comparisons
    n = prod(inputs[0].shape)
    num_operations = int(n * (n ** 0.5))  # Approximation for uniqueness complexity
    return num_operations


def bincount_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Bincount: count occurrences of each value
    # Each element requires a read + increment operation
    num_operations = 2 * prod(inputs[0].shape)
    return num_operations


def histc_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Histc: compute histogram
    # Each element requires a read + bin assignment operation
    num_operations = 2 * prod(inputs[0].shape)
    return num_operations


def mode_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Mode: find most frequent value
    # Each element requires a comparison operation
    # For n elements, approximately n*log(n) comparisons
    n = prod(inputs[0].shape)
    num_operations = int(n * (n ** 0.5))  # Approximation for mode complexity
    return num_operations


def median_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Median: find median value
    # Each element requires a comparison operation
    # For n elements, approximately n*log(n) comparisons
    n = prod(inputs[0].shape)
    num_operations = int(n * (n ** 0.5))  # Approximation for median complexity
    return num_operations


def quantile_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Quantile: find quantile value
    # Each element requires a comparison operation
    # For n elements, approximately n*log(n) comparisons
    n = prod(inputs[0].shape)
    num_operations = int(n * (n ** 0.5))  # Approximation for quantile complexity
    return num_operations


def std_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Std: compute standard deviation
    # Each element requires: read + square + sum + sqrt operations
    n = prod(inputs[0].shape)
    num_operations = 4 * n  # read + square + sum + sqrt
    return num_operations


def var_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Var: compute variance
    # Each element requires: read + square + sum operations
    n = prod(inputs[0].shape)
    num_operations = 3 * n  # read + square + sum
    return num_operations


def norm_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Norm: compute norm of tensor
    # Each element requires: read + square + sum + sqrt operations
    n = prod(inputs[0].shape)
    num_operations = 4 * n  # read + square + sum + sqrt
    return num_operations


def dist_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Dist: compute distance between tensors
    # Each element requires: read + subtract + square + sum + sqrt operations
    n = prod(inputs[0].shape)
    num_operations = 6 * n  # read + subtract + square + sum + sqrt
    return num_operations


def pdist_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Pdist: compute pairwise distances
    # Each pair requires: subtract + square + sum + sqrt operations
    n = prod(inputs[0].shape)
    num_operations = 5 * n  # subtract + square + sum + sqrt
    return num_operations


def cdist_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Cdist: compute cross distances
    # Each pair requires: subtract + square + sum + sqrt operations
    n = prod(inputs[0].shape)
    num_operations = 5 * n  # subtract + square + sum + sqrt
    return num_operations


def einsum_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Einsum: Einstein summation
    # This is a complex operation that depends on the equation
    # For now, use a conservative estimate
    equation = inputs[0] if isinstance(inputs[0], str) else ""
    if "->" in equation:
        # Extract output shape from equation
        output_shape = outputs[0].shape
        num_operations = prod(output_shape) * 10  # Conservative estimate
    else:
        # Fallback to basic operations
        num_operations = prod(outputs[0].shape)
    return num_operations


def baddbmm_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Baddbmm: batch add + batch matrix multiplication
    # Matrix multiplication + bias addition
    input_shapes = [v.shape for v in inputs[1:3]]
    batch_size = input_shapes[0][0]
    m, n = input_shapes[0][1], input_shapes[0][2]
    p = input_shapes[1][2]
    matmul_ops = batch_size * m * n * p
    bias_ops = batch_size * m * p
    num_operations = matmul_ops + bias_ops
    return num_operations


def addbmm_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Addbmm: add + batch matrix multiplication
    # Matrix multiplication + bias addition
    input_shapes = [v.shape for v in inputs[1:3]]
    batch_size = input_shapes[0][0]
    m, n = input_shapes[0][1], input_shapes[0][2]
    p = input_shapes[1][2]
    matmul_ops = batch_size * m * n * p
    bias_ops = m * p
    num_operations = matmul_ops + bias_ops
    return num_operations


def baddmm_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Baddmm: batch add + matrix multiplication
    # Matrix multiplication + bias addition
    input_shapes = [v.shape for v in inputs[1:3]]
    batch_size = input_shapes[0][0]
    m, n = input_shapes[0][1], input_shapes[0][2]
    p = input_shapes[1][1]
    matmul_ops = batch_size * m * n * p
    bias_ops = batch_size * m * p
    num_operations = matmul_ops + bias_ops
    return num_operations


def ger_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Ger: outer product of two vectors
    # Each output element requires a multiplication
    m = inputs[0].shape[0]
    n = inputs[1].shape[0]
    num_operations = m * n
    return num_operations


def outer_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Outer: outer product of two tensors
    # Each output element requires a multiplication
    num_operations = prod(outputs[0].shape)
    return num_operations


def cross_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Cross: cross product of two tensors
    # Each output element requires 3 multiplications + 2 subtractions
    num_operations = 5 * prod(outputs[0].shape)
    return num_operations


def dot_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Dot: dot product of two tensors
    # Each element requires a multiplication + addition
    n = prod(inputs[0].shape)
    num_operations = 2 * n  # multiplication + addition
    return num_operations


def solve_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Solve: solve linear system Ax = b
    # For n x n matrix, approximately n^3 operations
    n = inputs[0].shape[0]
    num_operations = n ** 3
    return num_operations


def inverse_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Inverse: compute matrix inverse
    # For n x n matrix, approximately n^3 operations
    n = inputs[0].shape[0]
    num_operations = n ** 3
    return num_operations


def pinverse_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Pinverse: compute pseudo-inverse
    # For m x n matrix, approximately max(m,n)^3 operations
    m, n = inputs[0].shape[:2]
    num_operations = max(m, n) ** 3
    return num_operations


def det_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Det: compute matrix determinant
    # For n x n matrix, approximately n^3 operations
    n = inputs[0].shape[0]
    num_operations = n ** 3
    return num_operations


def eig_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Eig: compute eigenvalues and eigenvectors
    # For n x n matrix, approximately n^3 operations
    n = inputs[0].shape[0]
    num_operations = n ** 3
    return num_operations


def svd_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # SVD: compute singular value decomposition
    # For m x n matrix, approximately max(m,n)^3 operations
    m, n = inputs[0].shape[:2]
    num_operations = max(m, n) ** 3
    return num_operations


def qr_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # QR: compute QR decomposition
    # For m x n matrix, approximately 2*m*n^2 operations
    m, n = inputs[0].shape[:2]
    num_operations = 2 * m * (n ** 2)
    return num_operations


def lu_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # LU: compute LU decomposition
    # For n x n matrix, approximately n^3 operations
    n = inputs[0].shape[0]
    num_operations = n ** 3
    return num_operations


def cholesky_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Cholesky: compute Cholesky decomposition
    # For n x n matrix, approximately n^3/3 operations
    n = inputs[0].shape[0]
    num_operations = (n ** 3) // 3
    return num_operations


def triangular_solve_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Triangular_solve: solve triangular system
    # For n x n matrix, approximately n^2 operations
    n = inputs[0].shape[0]
    num_operations = n ** 2
    return num_operations


def lstsq_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Lstsq: least squares solution
    # For m x n matrix, approximately 2*m*n^2 operations
    m, n = inputs[0].shape[:2]
    num_operations = 2 * m * (n ** 2)
    return num_operations


def matrix_power_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Matrix_power: compute matrix power
    # For n x n matrix and power k, approximately k*n^3 operations
    n = inputs[0].shape[0]
    k = inputs[1] if len(inputs) > 1 else 2
    num_operations = k * (n ** 3)
    return num_operations


def fft_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # FFT: Fast Fourier Transform
    # For n elements, approximately n*log(n) operations
    n = prod(inputs[0].shape)
    num_operations = int(n * (n ** 0.5))  # Approximation for FFT complexity
    return num_operations


def ifft_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # IFFT: Inverse Fast Fourier Transform
    # For n elements, approximately n*log(n) operations
    n = prod(inputs[0].shape)
    num_operations = int(n * (n ** 0.5))  # Approximation for IFFT complexity
    return num_operations


def stft_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # STFT: Short-Time Fourier Transform
    # For n elements, approximately n*log(n) operations
    n = prod(inputs[0].shape)
    num_operations = int(n * (n ** 0.5))  # Approximation for STFT complexity
    return num_operations


def istft_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # ISTFT: Inverse Short-Time Fourier Transform
    # For n elements, approximately n*log(n) operations
    n = prod(inputs[0].shape)
    num_operations = int(n * (n ** 0.5))  # Approximation for ISTFT complexity
    return num_operations


def conv1d_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Conv1d: 1D convolution
    # Use the existing convolution_ops function
    return convolution_ops(inputs, outputs)


def conv2d_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Conv2d: 2D convolution
    # Use the existing convolution_ops function
    return convolution_ops(inputs, outputs)


def conv3d_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Conv3d: 3D convolution
    # Use the existing convolution_ops function
    return convolution_ops(inputs, outputs)


def conv_transpose1d_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Conv_transpose1d: 1D transposed convolution
    # Use the existing convolution_ops function
    return convolution_ops(inputs, outputs)


def conv_transpose2d_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Conv_transpose2d: 2D transposed convolution
    # Use the existing convolution_ops function
    return convolution_ops(inputs, outputs)


def conv_transpose3d_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Conv_transpose3d: 3D transposed convolution
    # Use the existing convolution_ops function
    return convolution_ops(inputs, outputs)


def unfold_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Unfold: extract sliding local blocks
    # Each element requires a copy operation
    num_operations = prod(outputs[0].shape)
    return num_operations


def fold_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Fold: combine sliding local blocks
    # Each element requires a copy operation
    num_operations = prod(outputs[0].shape)
    return num_operations


def adaptive_avg_pool1d_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Adaptive_avg_pool1d: adaptive average pooling
    # Each output element requires averaging operations
    num_operations = prod(outputs[0].shape)
    return num_operations


def adaptive_max_pool1d_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Adaptive_max_pool1d: adaptive max pooling
    # Each output element requires comparison operations
    num_operations = prod(outputs[0].shape)
    return num_operations


def adaptive_avg_pool3d_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Adaptive_avg_pool3d: adaptive average pooling
    # Each output element requires averaging operations
    num_operations = prod(outputs[0].shape)
    return num_operations


def adaptive_max_pool3d_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Adaptive_max_pool3d: adaptive max pooling
    # Each output element requires comparison operations
    num_operations = prod(outputs[0].shape)
    return num_operations


def max_pool1d_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Max_pool1d: 1D max pooling
    # Each output element requires comparison operations
    num_operations = prod(outputs[0].shape)
    return num_operations


def max_pool3d_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Max_pool3d: 3D max pooling
    # Each output element requires comparison operations
    num_operations = prod(outputs[0].shape)
    return num_operations


def avg_pool1d_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Avg_pool1d: 1D average pooling
    # Each output element requires averaging operations
    num_operations = prod(outputs[0].shape)
    return num_operations


def avg_pool3d_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Avg_pool3d: 3D average pooling
    # Each output element requires averaging operations
    num_operations = prod(outputs[0].shape)
    return num_operations


def max_pool1d_with_indices_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Max_pool1d_with_indices: 1D max pooling with indices
    # Each output element requires comparison operations
    num_operations = prod(outputs[0].shape)
    return num_operations


def max_pool3d_with_indices_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Max_pool3d_with_indices: 3D max pooling with indices
    # Each output element requires comparison operations
    num_operations = prod(outputs[0].shape)
    return num_operations


def fractional_max_pool2d_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Fractional_max_pool2d: fractional max pooling
    # Each output element requires comparison operations
    num_operations = prod(outputs[0].shape)
    return num_operations


def fractional_max_pool3d_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Fractional_max_pool3d: fractional max pooling
    # Each output element requires comparison operations
    num_operations = prod(outputs[0].shape)
    return num_operations


def lp_pool1d_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Lp_pool1d: Lp norm pooling
    # Each output element requires power + sum + power operations
    num_operations = 3 * prod(outputs[0].shape)
    return num_operations


def lp_pool2d_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Lp_pool2d: Lp norm pooling
    # Each output element requires power + sum + power operations
    num_operations = 3 * prod(outputs[0].shape)
    return num_operations


def lp_pool3d_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Lp_pool3d: Lp norm pooling
    # Each output element requires power + sum + power operations
    num_operations = 3 * prod(outputs[0].shape)
    return num_operations


def adaptive_lp_pool1d_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Adaptive_lp_pool1d: adaptive Lp norm pooling
    # Each output element requires power + sum + power operations
    num_operations = 3 * prod(outputs[0].shape)
    return num_operations


def adaptive_lp_pool2d_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Adaptive_lp_pool2d: adaptive Lp norm pooling
    # Each output element requires power + sum + power operations
    num_operations = 3 * prod(outputs[0].shape)
    return num_operations


def adaptive_lp_pool3d_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Adaptive_lp_pool3d: adaptive Lp norm pooling
    # Each output element requires power + sum + power operations
    num_operations = 3 * prod(outputs[0].shape)
    return num_operations


def max_unpool1d_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Max_unpool1d: unpooling with indices
    # Each element requires a copy operation
    num_operations = prod(outputs[0].shape)
    return num_operations


def max_unpool2d_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Max_unpool2d: unpooling with indices
    # Each element requires a copy operation
    num_operations = prod(outputs[0].shape)
    return num_operations


def max_unpool3d_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Max_unpool3d: unpooling with indices
    # Each element requires a copy operation
    num_operations = prod(outputs[0].shape)
    return num_operations


def replication_pad1d_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Replication_pad1d: replication padding
    # Each padded element requires a copy operation
    input_size = prod(inputs[0].shape)
    output_size = prod(outputs[0].shape)
    num_operations = output_size - input_size
    return num_operations


def replication_pad2d_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Replication_pad2d: replication padding
    # Each padded element requires a copy operation
    input_size = prod(inputs[0].shape)
    output_size = prod(outputs[0].shape)
    num_operations = output_size - input_size
    return num_operations


def replication_pad3d_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Replication_pad3d: replication padding
    # Each padded element requires a copy operation
    input_size = prod(inputs[0].shape)
    output_size = prod(outputs[0].shape)
    num_operations = output_size - input_size
    return num_operations


def circular_pad1d_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Circular_pad1d: circular padding
    # Each padded element requires a copy operation
    input_size = prod(inputs[0].shape)
    output_size = prod(outputs[0].shape)
    num_operations = output_size - input_size
    return num_operations


def circular_pad2d_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Circular_pad2d: circular padding
    # Each padded element requires a copy operation
    input_size = prod(inputs[0].shape)
    output_size = prod(outputs[0].shape)
    num_operations = output_size - input_size
    return num_operations


def circular_pad3d_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Circular_pad3d: circular padding
    # Each padded element requires a copy operation
    input_size = prod(inputs[0].shape)
    output_size = prod(outputs[0].shape)
    num_operations = output_size - input_size
    return num_operations


def pad_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Pad: general padding
    # Each padded element requires a copy operation
    input_size = prod(inputs[0].shape)
    output_size = prod(outputs[0].shape)
    num_operations = output_size - input_size
    return num_operations


def interpolate_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    # Interpolate: resize tensor
    # Each output element requires interpolation operations
    num_operations = prod(outputs[0].shape)
    return num_operations


operations_mapping = {
    ops.aten.min: max_ops,
    ops.aten.max: max_ops,
    ops.aten.sort: basic_ops,
    ops.aten.eq: basic_ops,
    ops.aten.ne: basic_ops,
    ops.aten.add_: basic_ops,
    ops.aten.add: basic_ops,
    ops.aten.sub: basic_ops,
    ops.aten.mul: basic_ops,
    ops.aten.mul_: basic_ops,
    ops.aten.pow_: basic_ops,
    ops.aten.exp: basic_ops,
    ops.aten.sum: sum_ops,
    ops.aten.div: basic_ops,
    ops.aten.div_: basic_ops,
    ops.aten.bitwise_not: basic_ops,
    ops.aten.sin: basic_ops,
    ops.aten.sin_: basic_ops,
    ops.aten.cos: basic_ops,
    ops.aten.cos_: basic_ops,
    ops.aten.tan: basic_ops,
    ops.aten.tan_: basic_ops,
    ops.aten.sigmoid: basic_ops,
    ops.aten.sigmoid_: basic_ops,
    ops.aten.tanh: basic_ops,
    ops.aten.tanh_: basic_ops,
    ops.aten.neg: basic_ops,
    ops.aten.neg_: basic_ops,
    ops.aten.reciprocal: basic_ops,
    ops.aten.reciprocal_: basic_ops,
    ops.aten.abs: basic_ops,
    ops.aten.abs_: basic_ops,
    ops.aten.sqrt: basic_ops,
    ops.aten.sqrt_: basic_ops,
    ops.aten.gt: basic_ops,
    ops.aten.lt: basic_ops,
    ops.aten.ge: basic_ops,
    ops.aten.le: basic_ops,
    ops.aten.equal: basic_ops,
    ops.aten.rsqrt: basic_ops,
    ops.aten.rsqrt_: basic_ops,
    ops.aten.log: basic_ops,
    ops.aten.log_: basic_ops,
    ops.aten.log2: basic_ops,
    ops.aten.log2_: basic_ops,
    ops.aten.log10: basic_ops,
    ops.aten.log10_: basic_ops,
    ops.aten.minimum: basic_ops,
    ops.aten.maximum: basic_ops,
    ops.aten.floor: basic_ops,
    ops.aten.ceil: basic_ops,
    ops.aten.round: basic_ops,
    ops.aten.trunc: basic_ops,
    ops.aten.frac: basic_ops,
    ops.aten.erf: basic_ops,
    ops.aten.erfc: basic_ops,
    ops.aten.erfinv: basic_ops,
    ops.aten.digamma: basic_ops,
    ops.aten.polygamma: basic_ops,
    ops.aten.trunc: basic_ops,
    ops.aten.frac: basic_ops,
    ops.aten.lerp: basic_ops,
    ops.aten.clamp: basic_ops,
    ops.aten.clamp_: basic_ops,
    ops.aten.where: basic_ops,
    ops.aten.scatter: scatter_add_ops,
    ops.aten.scatter_: scatter_add_ops,
    ops.aten.scatter_add: scatter_add_ops,
    ops.aten.scatter_add_: scatter_add_ops,
    ops.aten.scatter_reduce: scatter_reduce_ops,
    ops.aten.scatter_reduce_: scatter_reduce_ops,
    ops.aten.arange: arange_ops,
    ops.aten.mean: mean_ops,
    ops.aten.mm: matmul_ops,
    ops.aten.matmul: matmul_ops,
    ops.aten.addmm: addmm_ops,
    ops.aten.bmm: bmm_ops,
    ops.aten.relu: relu_ops,
    ops.aten.relu_: relu_ops,
    ops.aten.leaky_relu: leaky_relu_ops,
    ops.aten.leaky_relu_: leaky_relu_ops,
    ops.aten.elu: leaky_relu_ops,
    ops.aten.elu_: leaky_relu_ops,
    ops.aten._softmax: softmax_ops,
    ops.aten._log_softmax: log_softmax_ops,
    ops.aten.native_batch_norm: native_batch_norm_ops,
    ops.aten.convolution: convolution_ops,
    ops.aten._convolution: convolution_ops,
    ops.aten.convolution_backward: convolution_backward_ops,
    ops.aten.max_pool2d_with_indices: max_pool2d_with_indices_ops,
    ops.aten.max_pool2d_with_indices_backward: max_pool2d_with_indices_backward_ops,
    ops.aten.avg_pool2d: avg_pool2d_ops,
    ops.aten.constant_pad_nd: constant_pad_nd_ops,
    ops.aten.native_batch_norm_backward: native_batch_norm_backward_ops,
    ops.aten.threshold_backward: threshold_backward_ops,
    ops.aten.gather: gather_ops,
    ops.aten.index_add: index_add_ops,
    ops.aten.index_add_: index_add_ops,
    ops.aten.new_zeros: new_zeros_ops,
    ops.aten.masked_fill: masked_fill_ops,
    ops.aten.masked_fill_: masked_fill_ops,
    ops.aten.full: full_ops,
    ops.aten.layer_norm: layer_norm_ops,
    ops.aten.dropout: dropout_ops,
    ops.aten.embedding: embedding_ops,
    ops.aten.lstm_cell: lstm_cell_ops,
    ops.aten.adaptive_avg_pool2d: avg_pool2d_ops,
    ops.aten.adaptive_max_pool2d: max_pool2d_with_indices_ops,
    ops.aten.dropout_: dropout_ops,
    ops.aten.group_norm: layer_norm_ops,
    ops.aten.instance_norm: layer_norm_ops,
    ops.aten.embedding_backward: basic_ops,
    ops.aten.gru_cell: lstm_cell_ops,
    ops.aten.rnn_tanh_cell: basic_ops,
    ops.aten.rnn_relu_cell: basic_ops,
    ops.aten.cat: cat_ops,
    ops.aten.split: split_ops,
    ops.aten.view: view_ops,
    ops.aten.reshape: view_ops,
    ops.aten.transpose: transpose_ops,
    ops.aten.permute: transpose_ops,
    ops.aten.linear: linear_ops,
    ops.aten.gelu: gelu_ops,
    ops.aten.unsqueeze: unsqueeze_ops,
    ops.aten.squeeze: squeeze_ops,
    ops.aten.expand: expand_ops,
    ops.aten.repeat: repeat_ops,
    ops.aten.stack: stack_ops,
    ops.aten.chunk: chunk_ops,
    ops.aten.narrow: narrow_ops,
    ops.aten.index_select: index_select_ops,
    ops.aten.masked_select: masked_select_ops,
    ops.aten.take: take_ops,
    ops.aten.put: put_ops,
    ops.aten.roll: roll_ops,
    ops.aten.flip: flip_ops,
    ops.aten.rot90: rot90_ops,
    ops.aten.diag: diag_ops,
    ops.aten.tril: tril_ops,
    ops.aten.triu: triu_ops,
    ops.aten.eye: eye_ops,
    ops.aten.ones: ones_ops,
    ops.aten.zeros: zeros_ops,
    ops.aten.randn: randn_ops,
    ops.aten.normal: normal_ops,
    ops.aten.uniform: uniform_ops,
    ops.aten.clone: clone_ops,
    ops.aten.detach: detach_ops,
    ops.aten.contiguous: contiguous_ops,
    ops.aten.to: to_ops,
    ops.aten.cpu: cpu_ops,
    ops.aten.cuda: cuda_ops,
    ops.aten.item: item_ops,
    ops.aten.numel: numel_ops,
    ops.aten.dim: dim_ops,
    ops.aten.size: size_ops,
    ops.aten.is_contiguous: is_contiguous_ops,
    ops.aten.device: device_ops,
    ops.aten.fill_: fill_ops,
    ops.aten.zero_: zero_ops,
    ops.aten.random_: random_ops,
    ops.aten.index: index_ops,
    ops.aten.index_copy_: index_copy_ops,
    ops.aten.index_fill_: index_fill_ops,
    ops.aten.index_add_: index_add_ops_advanced,
    ops.aten.masked_scatter: masked_scatter_ops,
    ops.aten.masked_select: masked_select_ops_advanced,
    ops.aten.masked_fill: masked_fill_ops_advanced,
    ops.aten.masked_scatter_: masked_scatter_ops_advanced,
    ops.aten.gather: gather_ops_advanced,
    ops.aten.scatter: scatter_ops_advanced,
    ops.aten.scatter_add: scatter_add_ops_advanced,
    ops.aten.scatter_reduce: scatter_reduce_ops_advanced,
    ops.aten.topk: topk_ops,
    ops.aten.sort: sort_ops_advanced,
    ops.aten.argsort: argsort_ops,
    ops.aten.kthvalue: kthvalue_ops,
    ops.aten.bincount: bincount_ops,
    ops.aten.histc: histc_ops,
    ops.aten.mode: mode_ops,
    ops.aten.median: median_ops,
    ops.aten.quantile: basic_ops,
    ops.aten.std: std_ops,
    ops.aten.var: var_ops,
    ops.aten.norm: norm_ops,
    ops.aten.dist: dist_ops,
    ops.aten.pdist: pdist_ops,
    ops.aten.cdist: cdist_ops,
    ops.aten.einsum: einsum_ops,
    ops.aten.baddbmm: baddbmm_ops,
    ops.aten.addbmm: addbmm_ops,
    ops.aten.ger: ger_ops,
    ops.aten.outer: outer_ops,
    ops.aten.cross: cross_ops,
    ops.aten.dot: dot_ops,
    ops.aten.inverse: inverse_ops,
    ops.aten.pinverse: pinverse_ops,
    ops.aten.det: det_ops,
    ops.aten.svd: svd_ops,
    ops.aten.qr: qr_ops,
    ops.aten.cholesky: cholesky_ops,
    ops.aten.triangular_solve: triangular_solve_ops,
    ops.aten.matrix_power: matrix_power_ops,
    ops.aten.conv1d: conv1d_ops,
    ops.aten.conv2d: conv2d_ops,
    ops.aten.conv3d: conv3d_ops,
    ops.aten.conv_transpose1d: conv_transpose1d_ops,
    ops.aten.conv_transpose2d: conv_transpose2d_ops,
    ops.aten.conv_transpose3d: conv_transpose3d_ops,
    ops.aten.unfold: unfold_ops,
    ops.aten.adaptive_avg_pool1d: adaptive_avg_pool1d_ops,
    ops.aten.adaptive_max_pool1d: adaptive_max_pool1d_ops,
    ops.aten.adaptive_avg_pool3d: adaptive_avg_pool3d_ops,
    ops.aten.adaptive_max_pool3d: adaptive_max_pool3d_ops,
    ops.aten.max_pool1d: max_pool1d_ops,
    ops.aten.max_pool3d: max_pool3d_ops,
    ops.aten.avg_pool1d: avg_pool1d_ops,
    ops.aten.avg_pool3d: avg_pool3d_ops,
    ops.aten.max_pool1d_with_indices: max_pool1d_with_indices_ops,
    ops.aten.max_pool3d_with_indices: max_pool3d_with_indices_ops,
    ops.aten.fractional_max_pool2d: fractional_max_pool2d_ops,
    ops.aten.fractional_max_pool3d: fractional_max_pool3d_ops,
    ops.aten.replication_pad1d: replication_pad1d_ops,
    ops.aten.replication_pad2d: replication_pad2d_ops,
    ops.aten.replication_pad3d: replication_pad3d_ops,
    ops.aten.pad: pad_ops,
    ops.aten.any: basic_ops,
    ops.aten.isnan: basic_ops,
    ops.aten.ceil_: basic_ops,
    ops.aten.lerp_: basic_ops,
    ops.aten.squeeze_: basic_ops,
    ops.aten._to_copy: basic_ops,
    ops.aten.scalar_tensor: basic_ops,
    ops.aten.pow: basic_ops,
    ops.aten.cudnn_batch_norm: native_batch_norm_ops,
    ops.aten.diagonal_copy: diag_ops,
    ops.aten.native_dropout: dropout_ops,
    ops.aten._fft_r2c: fft_ops,
    ops.aten.native_layer_norm: layer_norm_ops,
    ops.aten._thnn_fused_lstm_cell: lstm_cell_ops,
    ops.aten.linalg_cholesky_ex: cholesky_ops,
    ops.aten.linalg_vector_norm: norm_ops,
    ops.aten.reflection_pad1d: constant_pad_nd_ops,
    ops.aten.reflection_pad2d: constant_pad_nd_ops,
    ops.aten.reflection_pad3d: constant_pad_nd_ops,
    ops.aten._unique2: unique_ops,
    ops.aten.linalg_cholesky_ex: choleksy_ops,
    ops.aten.cholesky_inverse: cholesky_inverse_ops,
}
