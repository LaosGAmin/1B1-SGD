import torch


class LBL_SGD_Function(torch.autograd.Function):
    """
    LBL_SGD_Function is like gradient checkpoint, we rewrote some
    of it, and it also works very similar to gradient checkpoint
    """
    input_data = None
    COUNT = 0
    lr = None
    cuda_state = None
    @staticmethod
    def set_data(data):
        LBL_SGD_Function.input_data = data

    @staticmethod
    def set_lr(lr):
        LBL_SGD_Function.lr = lr

    @staticmethod
    def set_state(cuda_state):
        LBL_SGD_Function.cuda_state = cuda_state

    @staticmethod
    def forward(ctx, pre_functions, run_function, preserve_rng_state, *args):
        """

        :param ctx:
        :param pre_functions: nn.ModuleList
        :param run_function: nn.ModuleList
        :param args: inpuit->tensor
        :return: output->tensor
        """
        ctx.pre_functions = pre_functions
        ctx.run_function = run_function

        outputs = args[0]
        with torch.no_grad():
            for f in run_function:
                outputs = f(outputs)

        return outputs

    @staticmethod
    def backward(ctx, *args):
        torch.cuda.set_rng_state(LBL_SGD_Function.cuda_state)
        inputs = LBL_SGD_Function.input_data.clone()
        with torch.no_grad():
            if ctx.pre_functions:
                for f in ctx.pre_functions:
                    inputs = f(inputs)
            else:
                pass

        inputs.requires_grad = True
        outputs = inputs
        with torch.enable_grad():
            for f in ctx.run_function:
                outputs = f(outputs)

        torch.autograd.backward(outputs, args)
        for f in ctx.run_function:
            if hasattr(f, 'weight'):
                f.weight.data = f.weight.data - LBL_SGD_Function.lr * f.weight.grad
                f.weight.grad.zero_()
            if hasattr(f, 'bias'):
                f.bias.data = f.bias.data - LBL_SGD_Function.lr * f.bias.grad
                f.bias.grad.zero_()

        grads = inputs.grad
        return (None, None, None) + (grads,)


def LBLSGD(pre_functions, run_function, *args, use_reentrant: bool = True, **kwargs):
    """

    :param pre_functions: nn.ModuleList
    :param run_function: nn.ModuleList
    :param args: inpuit->tensor
    :param use_reentrant:
    :param kwargs:
    :return:
    """
    preserve = kwargs.pop('preserve_rng_state', True)
    if kwargs and use_reentrant:
        raise ValueError("Unexpected keyword arguments: " + ",".join(arg for arg in kwargs))
    if use_reentrant:
        return LBL_SGD_Function.apply(pre_functions, run_function, preserve, *args)
    else:
        return LBLSGD_without_reentrant(
            pre_functions,
            run_function,
            *args,
            **kwargs,
        )


def LBLSGD_without_reentrant(function, preserve_rng_state=True, *args, **kwargs):
    return None
