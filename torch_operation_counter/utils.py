from math import prod
from numbers import Number
from typing import List


def normalize_tuple(x):
    if not isinstance(x, tuple):
        return (x,)
    return x


def transpose_shape(shape):
    return [shape[1], shape[0]] + list(shape[2:])


def flatten_dictionary(d):
    for i in getattr(d, "values", lambda: d)():
        if isinstance(i, int):
            yield i
        elif i is not None:
            yield from flatten_dictionary(i)


def conv_ops_count(
    x_shape: List[int],
    w_shape: List[int],
    out_shape: List[int],
    transposed: bool = False,
) -> Number:
    """
    Count operations for convolution. Note only multiplication is
    counted. Computation for addition and bias is ignored.
    Operations for a transposed convolution are calculated as
    operations = (x_shape[2:] * prod(w_shape) * batch_size).
    Args:
        x_shape (list(int)): The input shape before convolution.
        w_shape (list(int)): The filter shape.
        out_shape (list(int)): The output shape after convolution.
        transposed (bool): is the convolution transposed
    Returns:
        int: the number of operations
    """
    batch_size = x_shape[0]
    conv_shape = (x_shape if transposed else out_shape)[2:]
    ops = batch_size * prod(w_shape) * prod(conv_shape)
    return ops
