"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        # Initialize weight parameter with Kaiming uniform distribution
        self.weight = Parameter(
            init.kaiming_uniform(
                fan_in=self.in_features,
                fan_out=self.out_features,
                requires_grad=True,
                device=device,
                dtype=dtype
            )
        )
        
        # Initialize bias if required
        if bias:
            bias_init = init.kaiming_uniform(
                fan_in=self.out_features,
                fan_out=1,
                requires_grad=True,
                device=device,
                dtype=dtype
            )
            self.bias = Parameter(bias_init.transpose())
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # Compute linear transformation
        output = X.matmul(self.weight)
        
        # Add bias if present
        if self.bias is not None:
            output = output + self.bias.broadcast_to(output.shape)
            
        return output
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        # Get batch size (first dimension)
        batch_size = X.shape[0]
        
        # Calculate total size of remaining dimensions
        feature_size = np.prod(X.shape[1:])
        
        # Reshape to (batch_size, flattened_features)
        return X.reshape((batch_size, feature_size))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # Process input through each module in sequence
        output = x
        for module in self.modules:
            output = module(output)
        return output
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        # Compute log sum of exponentials for normalization
        batch_size = logits.shape[0]
        num_classes = logits.shape[1]
        
        # Compute normalized probabilities in log space
        log_normalization = ops.logsumexp(logits, axes=1)
        
        # Convert labels to one-hot encoding
        y_one_hot = init.one_hot(
            num_classes=num_classes,
            y=y,
            device=y.device,
            dtype=y.dtype
        )
        
        # Compute log-likelihood of correct classes
        correct_class_logits = ops.summation(logits * y_one_hot, axes=1)
        
        # Return average negative log-likelihood
        return ops.summation(log_normalization - correct_class_logits) / batch_size
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        # Initialize learnable parameters
        self.weight = Parameter(
            init.ones(self.dim, requires_grad=True, device=device, dtype=dtype)
        )
        self.bias = Parameter(
            init.zeros(self.dim, requires_grad=True, device=device, dtype=dtype)
        )
        
        # Initialize running statistics
        self.running_mean = init.zeros(self.dim, device=device, dtype=dtype)
        self.running_var = init.ones(self.dim, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size = x.shape[0]
        feature_shape = (1, x.shape[1])
        
        if self.training:
            # Compute batch statistics
            batch_mean = x.sum((0,)) / batch_size
            batch_mean_broadcasted = batch_mean.reshape(feature_shape).broadcast_to(x.shape)
            
            # Compute variance
            centered_data = x - batch_mean_broadcasted
            batch_var = (centered_data ** 2).sum((0,)) / batch_size
            batch_var_broadcasted = batch_var.reshape(feature_shape).broadcast_to(x.shape)
            
            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean.data
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var.data
            
            # Normalize using batch statistics
            norm = centered_data / ((batch_var_broadcasted + self.eps) ** 0.5)
        else:
            # Use running statistics for inference
            running_mean_broadcasted = self.running_mean.reshape(feature_shape).broadcast_to(x.shape)
            running_var_broadcasted = self.running_var.reshape(feature_shape).broadcast_to(x.shape)
            norm = (x - running_mean_broadcasted) / ((running_var_broadcasted + self.eps) ** 0.5)
        
        # Apply scale and shift
        return (self.weight.reshape(feature_shape).broadcast_to(x.shape) * norm + 
                self.bias.reshape(feature_shape).broadcast_to(x.shape))
        ### END YOUR SOLUTION

class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        # Initialize learnable parameters
        self.weight = Parameter(init.ones(self.dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(self.dim, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # Calculate statistics along feature dimension
        feature_mean = (x.sum((1,)) / x.shape[1]).reshape((x.shape[0], 1)).broadcast_to(x.shape)
        
        # Compute variance
        centered_data = x - feature_mean
        feature_var = ((centered_data ** 2).sum((1,)) / x.shape[1]
                      ).reshape((x.shape[0], 1)).broadcast_to(x.shape)
        
        # Normalize
        normalized = centered_data / ((feature_var + self.eps) ** 0.5)
        
        # Apply scale and shift
        return (self.weight.reshape((1, self.dim)).broadcast_to(x.shape) * normalized + 
                self.bias.reshape((1, self.dim)).broadcast_to(x.shape))
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        ### BEGIN YOUR SOLUTION
        if self.training:
            # Create binary mask with probability (1-p) of keeping values
            dropout_mask = init.randb(
                *x.shape,
                p=1-self.p,
                device=x.device,
                dtype=x.dtype
            )
            
            # Apply mask and scale by dropout probability
            return x * dropout_mask / (1 - self.p)
        return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return x + self.fn(x)
        ### END YOUR SOLUTION
