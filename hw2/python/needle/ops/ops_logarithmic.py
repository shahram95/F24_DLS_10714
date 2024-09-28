from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_Z = Z.max(axis=-1, keepdims=True)
        exp_Z = array_api.exp(Z - max_Z)
        sum_exp_Z = exp_Z.sum(axis=-1, keepdims=True)
        return Z - max_Z - array_api.log(sum_exp_Z)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        softmax = exp(self(Z))
        sum_out_grad = summation(out_grad, axes=(-1,))
        return out_grad - softmax * reshape(sum_out_grad, sum_out_grad.shape + (1,))
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_Z = array_api.max(Z, axis=self.axes, keepdims=True)
        return array_api.log(array_api.sum(array_api.exp(Z - max_Z), axis=self.axes)) + array_api.squeeze(max_Z, axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        exp_Z = exp(Z - Z.realize_cached_data().max(axis=self.axes, keepdims=True))
        sum_exp_Z = summation(exp_Z, axes=self.axes)
        grad = out_grad / sum_exp_Z

        if self.axes is None:
            return grad.broadcast_to(Z.shape) * exp_Z
        grad_shape = [1 if i in self.axes else s for i, s in enumerate(Z.shape)]
        return grad.reshape(grad_shape).broadcast_to(Z.shape) * exp_Z
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

