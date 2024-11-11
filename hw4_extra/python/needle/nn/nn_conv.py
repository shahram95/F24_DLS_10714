"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        # Initialize convolutional kernel weights using Kaiming initialization
        fan_in = in_channels * kernel_size * kernel_size
        self.weight = Parameter(
            init.kaiming_uniform(
                fan_in=fan_in,
                fan_out=out_channels,
                shape=(kernel_size, kernel_size, in_channels, out_channels),
                dtype=dtype,
                device=device,
                requires_grad=True
            )
        )

        # Initialize bias if needed
        if bias:
            # Calculate bound for uniform initialization
            bias_bound = 1.0 / (fan_in)**0.5
            self.bias = Parameter(
                init.rand(
                    self.out_channels,
                    low=-bias_bound,
                    high=bias_bound,
                    device=device
                )
            )
        else:
            self.bias = None

        # Calculate padding size for 'same' padding
        self.padding = (kernel_size - 1) // 2
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # Convert from NCHW to NHWC format for convolution
        nhwc_input = x.transpose((1, 2)).transpose((2, 3))
        
        # Perform convolution operation
        conv_output = ops.conv(
            nhwc_input,
            self.weight,
            stride=self.stride,
            padding=self.padding
        )
        
        # Add bias if present
        if self.bias is not None:
            # Reshape and broadcast bias to match output dimensions
            bias_shaped = self.bias.reshape((1, 1, 1, self.out_channels))
            conv_output = conv_output + bias_shaped.broadcast_to(conv_output.shape)
        
        # Convert back to NCHW format
        nchw_output = conv_output.transpose((2, 3)).transpose((1, 2))
        return nchw_output
        ### END YOUR SOLUTION