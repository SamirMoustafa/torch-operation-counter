import inspect
from collections import OrderedDict

from torch import is_tensor
from torch.nn import Module

msg_special_args = {"edge_index", "edge_index_i", "edge_index_j", "size", "size_i", "size_j"}
aggr_special_args = {"index", "dim_size"}


def __process_size__(size):
    if isinstance(size, int):
        return [size, size]
    if is_tensor(size):
        return size.tolist()
    return list(size) if size else [None, None]


def __distribute__(params, kwargs):
    return {key: kwargs.get(key, param.default) for key, param in params.items()}


class MessagePassing(Module):
    def __init__(self, aggr="sum", flow: str = "source_to_target", node_dim: int = 0):
        super(MessagePassing, self).__init__()

        assert aggr in ["sum", "prod", "mean", "amax", "amin"], f"{aggr} is not a valid aggregation method."
        assert flow in ["source_to_target", "target_to_source"], f"{flow} is not a valid flow direction."
        assert node_dim >= 0 and isinstance(node_dim, int), "node_dim must be non-negative integer."

        self.aggr = aggr
        self.flow = flow
        self.node_dim = node_dim

        self.__msg_params__ = OrderedDict(inspect.signature(self.message).parameters)
        self.__aggr_params__ = OrderedDict(inspect.signature(self.aggregate).parameters)
        self.__aggr_params__.popitem(last=False)

        msg_args = set(self.__msg_params__.keys()) - msg_special_args
        aggr_args = set(self.__aggr_params__.keys()) - aggr_special_args
        self.__args__ = msg_args.union(aggr_args)

    def __collect__(self, edge_index, size, kwargs):
        i, j = (0, 1) if self.flow == "target_to_source" else (1, 0)
        ij = {"_i": i, "_j": j}

        out = {}
        for arg in self.__args__:
            idx = ij.get(arg[-2:])
            data = kwargs.get(arg[:-2] if idx is not None else arg)

            if idx is not None and is_tensor(data):
                size[idx] = data.shape[self.node_dim]
                out[arg] = data.index_select(self.node_dim, edge_index[idx])
            else:
                out[arg] = data

        size[0] = size[1] if size[0] is None else size[0]
        size[1] = size[0] if size[1] is None else size[1]

        out.update({"edge_index": edge_index,
                    "edge_index_i": edge_index[i],
                    "edge_index_j": edge_index[j],
                    "size": size,
                    "size_i": size[i],
                    "size_j": size[j],
                    "index": edge_index[i],
                    "dim_size": size[i]})
        return out

    def forward(self, edge_index, size=None, **kwargs):
        size = __process_size__(size)
        kwargs = self.__collect__(edge_index, size, kwargs)
        out = self.message(**__distribute__(self.__msg_params__, kwargs))
        out = self.aggregate(out, **__distribute__(self.__aggr_params__, kwargs))
        return out if not isinstance(out, tuple) else out[0]

    def aggregate(self, inputs, index, dim_size):
        num_features = inputs.shape[1]
        x = zeros((dim_size, num_features), dtype=inputs.dtype)
        x.scatter_reduce_(self.node_dim, index.view(-1, 1).expand(-1, num_features), inputs, self.aggr)
        return x

    def __repr__(self):
        return "{}(dtype={})".format(self.__class__.__name__, self.dtype)

    def message(self, x_j, edge_weight=None):
        raise NotImplementedError("Not implemented yet.")


if __name__ == "__main__":
    from torch import ones, rand, long, zeros, device

    from torch_ops_counter import OperationsCounterMode


    class NativeMessagePassing(MessagePassing):
        def message(self, x_j, edge_weight=None):
            return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    device = device("cpu")

    # Reddit dataset settings
    n = 232_965
    e = 114_615_892
    num_feature = 602
    out_feature = 41

    # # Cora dataset settings
    # n = 2_708
    # e = 5_278
    # num_feature = 1_433
    # out_feature = 7

    # # Pubmed dataset settings
    # n = 19_717
    # e = 44_324
    # num_feature = 500
    # out_feature = 3

    x = rand(n, num_feature).to(device)
    edge_index = ones((2, e), dtype=long).to(device)
    edge_weight = rand(e).to(device)

    conv = NativeMessagePassing()

    with OperationsCounterMode() as ops_counter:
        conv(edge_index, size=(n, n), x=x, edge_weight=edge_weight)

    counter_dict = ops_counter.operations_count
    for module, operation in counter_dict.items():
        for op_name, op_count in operation.items():
            print(f"{module}::{op_name}: {op_count} OPs")

    print()
    print(f"Total operations: {ops_counter.total_operations} OP(s)")
    print(f"Total parameters: {sum([p.numel() for p in [*conv.parameters()][:-1]])} Param(s)")

