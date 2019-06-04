import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair

def linear_fa_backward_hook(module, grad_input, grad_output):
    if grad_input[1] is not None:
        grad_input_fa = grad_output[0].mm(module.weight_fa)
        
        return (grad_input[0], grad_input_fa) + grad_input[2:]

def conv2d_fa_backward_hook(module, grad_input, grad_output):
    if grad_input[0] is not None:
        grad_input_fa = torch.nn.grad.conv2d_input(grad_input[0].size(), module.weight_fa, grad_output[0], stride=module.stride, padding=module.padding, dilation=module.dilation, groups=module.groups)
        
        return (grad_input_fa,) + grad_input[1:]

class LinearFA(nn.Module):
    """
    Implementation of a linear module which uses random feedback weights
    in its backward pass, as described in Lillicrap et al., 2016:

    https://www.nature.com/articles/ncomms13276
    """

    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True):
        super(LinearFA, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.weight_fa = Variable(torch.FloatTensor(out_features, in_features), requires_grad=False)
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.register_backward_hook(linear_fa_backward_hook)

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_fa, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class _ConvNdFA(nn.Module):
    """
    Implementation of an N-dimensional convolution module which uses random feedback weights
    in its backward pass, as described in Lillicrap et al., 2016:

    https://www.nature.com/articles/ncomms13276

    This code is exactly copied from the _ConvNd module in PyTorch, with the addition
    of the random feedback weights.
    """

    __constants__ = ['stride', 'padding', 'dilation', 'groups', 'bias', 'padding_mode']

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding,
                 groups, bias, padding_mode):
        super(_ConvNdFA, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        if transposed:
            self.weight = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))

            self.weight_fa = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size), requires_grad=False)
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))

            self.weight_fa = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size), requires_grad=False)
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_fa, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)

class Conv2dFA(_ConvNdFA):
    """
    Implementation of a 2D convolution module which uses random feedback weights
    in its backward pass, as described in Lillicrap et al., 2016:

    https://www.nature.com/articles/ncomms13276
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2dFA, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

        self.register_backward_hook(conv2d_fa_backward_hook)

    def forward(self, input):
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                            self.weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
