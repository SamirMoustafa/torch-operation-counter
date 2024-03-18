__all__ = ["aten_ignore"]

ignore = [
    "_local_scalar_dense",
    "_reshape_alias",
    "argmax",
    "bernoulli_",
    "cat",
    "clone",
    "copy_",
    "detach",
    "dropout",
    "empty",
    "empty_like",
    "expand",
    "expand_as",
    "fill_",
    "flatten",
    "full",
    "full_like",
    "getattr",
    "getitem",
    "index",
    "index_put_",
    "index_select",
    "masked_fill_",
    "new_empty",
    "new_empty",
    "new_full",
    "new_zeros",
    "ones",
    "ones_like",
    "permute",
    "repeat",
    "select",
    "sign",
    "size",
    "slice",
    "squeeze",
    "squeeze",
    "t",
    "type_as",
    "unsqueeze",
    "view",
    "view_as",
    "zeros",
    "zeros_like",
]


aten_ignore = [f"aten.{ignore_i}" for ignore_i in ignore]
