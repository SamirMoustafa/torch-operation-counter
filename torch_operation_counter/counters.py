from math import prod
from numbers import Number
from typing import Any, List

from torch import ops

from torch_operation_counter.utils import conv_ops_count, transpose_shape


def basic_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    num_operations = prod(inputs[0].shape)
    return num_operations


def arange_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    num_operations = len(outputs[0])
    return num_operations


def scatter_add_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    dim = inputs[1]
    index = inputs[2]
    src = inputs[3]
    assert index.dim() == src.dim(), "index and src must have the same number of dimensions"
    num_operations = 0
    if dim < 0:
        dim += src.dim()
    for d in range(src.dim()):
        if d == dim:
            unique_indices, counts = index.select(dim, d).view(-1).unique(return_counts=True)
            num_operations += counts.sum().item()
        else:
            num_operations += src[d].numel()
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
    num_operations = prod(input_shapes[0]) * input_shapes[-1][-1]
    return num_operations


def addmm_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    input_shapes = [v.shape for v in inputs[1:3]]
    batch_size, input_dim = input_shapes[0]
    output_dim = input_shapes[1][1]
    num_operations = batch_size * input_dim * output_dim
    return num_operations


def bmm_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    input_shapes = [v.shape for v in inputs]
    n, c, t = input_shapes[0]
    d = input_shapes[-1][-1]
    num_operations = n * c * t * d
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
    dims = inputs[1]
    num_operations = prod([inputs[0].size(dim) for dim in dims]) + len(dims)
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
    comparisons_per_window = (prod(kernel_size) - 1) * nC
    num_operations = comparisons_per_window * nC * nH * nW
    return num_operations


def constant_pad_nd_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    num_operations = len(inputs[1]) * prod(inputs[0].shape)
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
    num_operations = (mask_tensor == True).sum().item()
    return num_operations


def full_ops(inputs: List[Any], outputs: List[Any]) -> Number:
    output_tensor = outputs[0]
    num_operations = prod(output_tensor.shape)  # Each element set once
    return num_operations


operations_mapping = {
    ops.aten.min: basic_ops,
    ops.aten.max: basic_ops,
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
    ops.aten.sum: basic_ops,
    ops.aten.div: basic_ops,
    ops.aten.div_: basic_ops,
    ops.aten.bitwise_not: basic_ops,
    # None-basic operations
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
}
