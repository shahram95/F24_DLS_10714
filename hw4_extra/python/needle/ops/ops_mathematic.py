"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a**b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # Validate input types
        if not isinstance(node.inputs[0], NDArray) or not isinstance(node.inputs[1], NDArray):
            raise ValueError("Both inputs must be tensors (NDArray).")

        # Get base and exponent tensors
        base, exponent = node.inputs[0], node.inputs[1]
        
        # Compute gradients using power rule and chain rule
        # d/da (a^b) = b * a^(b-1)
        grad_base = out_grad * exponent * (base ** (exponent - 1))
        
        # d/db (a^b) = a^b * ln(a)
        grad_exponent = out_grad * (base**exponent) * log(base)
        
        return grad_base, grad_exponent
        ### END YOUR SOLUTION


def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # Compute gradient using power rule: d/dx(x^n) = n * x^(n-1)
        input_tensor = node.inputs[0]
        return out_grad * self.scalar * (input_tensor ** (self.scalar - 1))
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # Get the input nodes
        lhs, rhs = node.inputs
        
        # Calculate gradients with respect to both inputs
        grad_a = out_grad / rhs  # Gradient with respect to a
        grad_b = -out_grad * lhs / (rhs * rhs)  # Gradient with respect to b
        
        return grad_a, grad_b
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / self.scalar 
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes:
            x, y = self.axes[0], self.axes[1]
        else:
            # Default behavior: transpose last two dimensions
            x, y = a.ndim - 1, a.ndim - 2
            
        # Create list of axes and swap the specified dimensions
        permute_axes = list(range(a.ndim))
        permute_axes[x], permute_axes[y] = y, x
        
        # Perform the transpose operation using permute
        return a.permute(permute_axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad.transpose(self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.compact().reshape(self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input, = node.inputs
        return reshape(out_grad, input.shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # If shapes match, no broadcast needed
        if a.shape == self.shape:
            return a
        # Broadcast and ensure result is compact
        return array_api.broadcast_to(a, self.shape).compact()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        origin_shape = node.inputs[0].shape
        # If shapes match, no reduction needed
        if origin_shape == self.shape:
            return out_grad

        # Find dimensions that were broadcasted
        shrink_dims = [i for i in range(len(self.shape))]
        # Iterate from back to handle different length shapes
        for i, (ori, cur) in enumerate(zip(reversed(origin_shape), reversed(self.shape))):
            if ori == cur:
                shrink_dims[len(self.shape) - i - 1] = -1
        # Filter out non-broadcasted dimensions
        shrink_dims = tuple(filter(lambda x: x >= 0, shrink_dims))
        assert len(shrink_dims) > 0

        # Sum over broadcasted dimensions and reshape to original
        return out_grad.sum(shrink_dims).reshape(origin_shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # Handle multiple axes case
        if isinstance(self.axes, (list, tuple)) and len(self.axes) > 1:
            # Sum over each axis in reverse sorted order
            for axis in reversed(sorted(self.axes)):
                a = a.sum(axis=axis)
            return a
        # Single axis case
        return array_api.sum(a, axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input, = node.inputs
        input_shape = input.shape
        
        # Prepare shape for expansion
        expand_dims = list(input_shape)
        
        # Determine axes for summation
        if self.axes is None:
            # Sum over all dimensions
            axes = list(range(len(input_shape)))
        else:
            # Convert single axis to list if needed
            axes = [self.axes] if isinstance(self.axes, int) else self.axes
            
        # Set summed dimensions to 1
        for i in range(len(axes)):
            expand_dims[axes[i]] = 1
            
        # Reshape and broadcast back to original shape
        out_grad = reshape(out_grad, expand_dims)
        return broadcast_to(out_grad, input_shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        # Calculate gradients for both inputs
        lgrad = matmul(out_grad, rhs.transpose())
        rgrad = matmul(lhs.transpose(), out_grad)
        
        # Handle case where input dimensions are less than gradient dimensions
        if len(lhs.shape) < len(lgrad.shape):
            lgrad = lgrad.sum(tuple([i for i in range(len(lgrad.shape) - len(lhs.shape))]))
        if len(rhs.shape) < len(rgrad.shape):
            rgrad = rgrad.sum(tuple([i for i in range(len(rgrad.shape) - len(rhs.shape))]))
            
        return lgrad, rgrad
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / node.inputs[0]
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * exp(node.inputs[0])
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        out = node.realize_cached_data()
        return out_grad * Tensor(out > 0, device=out_grad.device)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_data = node.inputs[0].realize_cached_data()
        return out_grad * (1 - array_api.tanh(input_data) ** 2)
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        ### BEGIN YOUR SOLUTION
         # Validate input
        assert len(args) > 0, "Stack needs at least one array!"
        shape = args[0].shape
        for a in args:
            assert shape == a.shape, "All arrays need to be of the same size!"
            
        # Create output array
        n = len(args)
        new_shape = list(shape)
        new_shape.insert(self.axis, n)
        out = array_api.empty(new_shape, device=args[0].device)
        
        # Fill output array
        slices = [slice(0, s) for s in new_shape]
        for i, arr in enumerate(args):
            slices[self.axis] = slice(i, i + 1)
            out[tuple(slices)] = arr
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # Split gradient along stacking axis
        return split(out_grad, self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        # Get number of splits and new shape
        n = A.shape[self.axis]
        new_shape = list(A.shape)
        new_shape.pop(self.axis)
        
        # Create slices for each split
        slices = [slice(0, s) for s in A.shape]
        splits = []
        
        # Split tensor along axis
        for i in range(n):
            slices[self.axis] = slice(i, i+1)
            splits.append(A[tuple(slices)].compact().reshape(new_shape))
        return tuple(splits)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # Stack gradients along split axis
        return stack(out_grad, self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.flip(a, self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad, self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # Calculate new shape with dilation
        new_shape = list(a.shape)
        slices = [slice(None)] * len(a.shape)
        
        # Prepare slicing and expand dimensions for dilation
        for axis in self.axes:
            new_shape[axis] *= (self.dilation + 1)
            slices[axis] = slice(None, None, self.dilation + 1)
        
        # Create dilated tensor filled with zeros
        out = array_api.full(new_shape, 0.0, dtype=a.dtype, device=a.device)
        # Insert original values at strided positions
        out[tuple(slices)] = a
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # Create slices to extract original elements
        slices = [slice(None)] * len(a.shape)
        for axis in self.axes:
            slices[axis] = slice(None, None, self.dilation + 1)
        
        # Extract elements at strided positions
        return a[tuple(slices)]
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        # Apply padding to input tensor
        pad_axes = [(0, 0)] + [(self.padding, self.padding)] * (A.ndim - 2) + [(0, 0)]
        A = A.pad(pad_axes)
        
        # Get dimensions
        N, H, W, C_in = A.shape
        K, _, _, C_out = B.shape
        Ns, Hs, Ws, Cs = A.strides
        
        # Reshape input for matrix multiplication
        inner_dim = K * K * C_in
        A = A.as_strided(
            shape=(N, H-K+1, W-K+1, K, K, C_in),
            strides=(Ns, Hs, Ws, Hs, Ws, Cs)
        ).compact()
        
        # Prepare matrices for multiplication
        A = A.reshape((N * (H-K+1) * (W-K+1), inner_dim))
        B = B.compact()
        
        # Perform convolution as matrix multiplication
        out = A @ B.reshape((inner_dim, C_out))
        out = out.reshape((N, H-K+1, W-K+1, C_out))
        
        # Apply striding if needed
        if self.stride > 1:
            slices = [slice(None)] * len(out.shape)
            slices[1] = slice(None, None, self.stride)  # H-dimension
            slices[2] = slice(None, None, self.stride)  # W-dimension
            out = out[tuple(slices)]
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        A, B = node.inputs
        N, H, W, C_in = A.shape
        K, _, _, C_out = B.shape
        
        # Handle strided convolution gradient
        if self.stride > 1:
            out_grad = dilate(out_grad, (1, 2), dilation=self.stride - 1)
            
        # Gradient with respect to input A
        # Flip kernel and transpose for input gradient computation
        B_t = flip(B, (0, 1)).transpose((2, 3))
        A_grad = conv(out_grad, B_t, padding=K - 1 - self.padding)
        
        # Gradient with respect to kernel B
        # Transpose input and output gradient for kernel gradient computation
        A_t = A.transpose((0, 3))
        out_grad_t = out_grad.transpose((0, 2)).transpose((0, 1))
        B_grad_t = conv(A_t, out_grad_t, padding=self.padding)
        B_grad = B_grad_t.transpose((0, 2)).transpose((0, 1))
        
        return A_grad, B_grad
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)


