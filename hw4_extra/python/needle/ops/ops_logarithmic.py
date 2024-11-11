from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        # Compute max for numerical stability
        max_values = Z.max(self.axes, keepdims=True)
        reduced_max = Z.max(self.axes)
        
        # Compute log(sum(exp(x))) with the max subtracted for stability
        shifted_exp = array_api.exp(Z - max_values.broadcast_to(Z.shape))
        summed_exp = array_api.sum(shifted_exp, self.axes)
        
        # Add back the max values
        return array_api.log(summed_exp) + reduced_max
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # Get input tensor
        input_tensor = node.inputs[0]
        
        # Compute max values for stable exp
        max_values = Tensor(
            input_tensor.realize_cached_data().max(axis=self.axes, keepdims=True),
            device=input_tensor.device
        )
        
        # Compute exp(x - max(x))
        shifted_tensor = input_tensor - max_values.broadcast_to(input_tensor.shape)
        exp_values = exp(shifted_tensor)
        
        # Compute sum(exp(x - max(x)))
        sum_exp = summation(exp_values, self.axes)
        
        # Compute gradient contribution
        grad_sum = out_grad / sum_exp
        
        # Prepare shape for broadcasting
        expanded_shape = list(input_tensor.shape)
        axes = (range(len(expanded_shape)) if self.axes is None 
               else (self.axes,) if isinstance(self.axes, Number) 
               else self.axes)
        
        for axis in axes:
            expanded_shape[axis] = 1
            
        # Broadcast gradient
        broadcasted_grad = grad_sum.reshape(expanded_shape).broadcast_to(input_tensor.shape)
        
        return broadcasted_grad * exp_values
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

