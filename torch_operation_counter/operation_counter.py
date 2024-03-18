import os
os.environ.update({"KINETO_LOG_LEVEL": "99"})

from collections import defaultdict
from operator import itemgetter
from warnings import warn

from torch import Tensor
from torch.nn import Module
from torch.autograd import Function
from torch.utils._pytree import tree_map
from torch.utils._python_dispatch import TorchDispatchMode

from torch_operation_counter.counters import operations_mapping
from torch_operation_counter.ignore_modules import aten_ignore
from torch_operation_counter.utils import normalize_tuple

MAIN_NAME = "__main__"


class OperationsCounterMode(TorchDispatchMode):
    """
    A class to count the number of operations in a PyTorch model. This class inherits from TorchDispatchMode,
    which is a context manager for registering hooks for forward and backward methods of PyTorch modules.

    Attributes:
    -----------
    ops_count: dict
        A dictionary that counts the number of operations for each function.
    total_main_operation: int
        The total number of operations for the main function.
    """

    def __init__(self, module: Module = None):
        """
        :param module: nn.Module, module to be registered for counting operations.
        """
        super().__init__()
        self.operations_count = defaultdict(lambda: defaultdict(int))
        self.total_main_operation = None
        self.parents = [MAIN_NAME]
        if module is not None:
            for name, module in dict(module.named_children()).items():
                module.register_forward_pre_hook(self.enter_module(name))
                module.register_forward_hook(self.exit_module(name))

    def enter_module(self, name):
        """
        Registers a hook for the forward pre-hook of a module.

        :param name: The name of the PyTorch module.
        :return:The forward pre-hook function.
        """

        def f(module, x):
            self.parents.append(name)
            x = normalize_tuple(x)
            x = self.create_backwards_pop(name)(*x)
            return x

        return f

    def exit_module(self, name):
        """
        Registers a hook for the forward hook of a PyTorch module.

        :param name: The name of the module.
        :return: The forward hook function.
        """

        def f(module, x, y):
            assert self.parents[-1] == name
            self.parents.pop()
            y = normalize_tuple(y)
            return self.create_backwards_push(name)(*y)

        return f

    def create_backwards_push(self, name):
        """
        Creates a function for the backwards push.

        :param name: The name of the PyTorch module.
        :return: The function for the backwards push.
        """

        class PushState(Function):
            @staticmethod
            def forward(ctx, *args):
                args = tree_map(lambda x: x.clone() if isinstance(x, Tensor) else x, args)
                if len(args) == 1:
                    return args[0]
                return args

            @staticmethod
            def backward(ctx, *grad_outs):
                self.parents.append(name)
                return grad_outs

        return PushState.apply

    def create_backwards_pop(self, name):
        """
        Creates a function to be used in the backward pass of the model.

        :param name: name of the module from which it was called. This function is used to keep track of the nested
                     module hierarchy when computing the operations count.
        :return: function applies the gradient of the output to the input by calling the backward()
        """

        class PopState(Function):
            @staticmethod
            def forward(ctx, *args):
                args = tree_map(lambda x: x.clone() if isinstance(x, Tensor) else x, args)
                if len(args) == 1:
                    return args[0]
                return args

            @staticmethod
            def backward(ctx, *grad_outs):
                assert self.parents[-1] == name
                self.parents.pop()
                return grad_outs

        return PopState.apply

    def __enter__(self):
        """
        This method is called when the with block for the OperationsCounterMode object is entered. It clears the
        ops_count dictionary and calls the parent __enter__() method.
        """
        self.operations_count.clear()
        super().__enter__()
        return self

    def __exit__(self, *args):
        """
        This method is called when the with block for the OperationsCounterMode object is exited. It sorts the ops_count
        dictionary in descending order by the number of operations counted and calls the parent __exit__() method.
        """
        for mod in self.operations_count.keys():
            operations_count_mod_items = self.operations_count[mod].items()
            self.operations_count[mod] = dict(sorted(operations_count_mod_items, key=itemgetter(1), reverse=True))
        super().__exit__(*args)

    @property
    def total_operations(self):
        """
        total number of operations counted in the forward pass of the model. It does this by summing the number of
        operations for each module in the ops_count dictionary.
        """
        assert self.total_main_operation is not None, "No operations were computed to count."
        return self.total_main_operation

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        """
        This method is called when the OperationsCounterMode object is used as a dispatch mode in a PyTorch function
        call. It updates the ops_count dictionary with the number of operations for the given function call. It also
        counts operations using the PyTorch profiler if the function call involves GPU computations.

        :param func:  The PyTorch function being called
        :param types: Tuple of argument types
        :param args: Tuple of input arguments
        :param kwargs: The dictionary of keyword argument
        :return: The output of the PyTorch function call
        """
        kwargs = kwargs if kwargs else {}
        operation_count = 0
        out = func(*args, **kwargs)
        func_packet = func._overloadpacket
        for par in self.parents:
            self.operations_count[par][func_packet] += operation_count
        if operation_count == 0:
            if func_packet in operations_mapping:
                operation_count = operations_mapping[func_packet](args, normalize_tuple(out))
                for par in self.parents:
                    self.operations_count[par][func_packet] += operation_count
            else:
                if str(func_packet) not in aten_ignore:
                    warn(f"operation `{func_packet}` is not considered in counting OPs.")

        self.total_main_operation = sum(self.operations_count[MAIN_NAME].values())

        return out
