import math
from .init_basic import *


def xavier_uniform(fan_in, fan_out, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    # Calculate shape based on input parameters
    if shape := kwargs.get('shape'):
        fan_in = math.prod(shape[:-1])
    else:
        shape = (fan_in, fan_out)
    
    # Calculate bound using Xavier/Glorot formula
    bound = gain * math.sqrt(6.0 / (fan_in + fan_out))
    
    # Generate uniform distribution within calculated bounds
    return rand(*shape, low=-bound, high=bound, **kwargs)
    ### END YOUR SOLUTION


def xavier_normal(fan_in, fan_out, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    # Calculate shape based on input parameters
    if shape := kwargs.get('shape'):
        fan_in = math.prod(shape[:-1])
    else:
        shape = (fan_in, fan_out)
    
    # Calculate standard deviation using Xavier/Glorot formula
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    
    # Generate normal distribution with calculated std
    return randn(*shape, mean=0.0, std=std, **kwargs)
    ### END YOUR SOLUTION



def kaiming_uniform(fan_in, fan_out, shape=None, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    # Calculate effective fan_in from shape if provided
    if shape is not None:
        fan_in = math.prod(shape[:-1])
    else:
        shape = (fan_in, fan_out)
    
    # Calculate bound using He initialization formula
    bound = math.sqrt(2.0) * math.sqrt(3.0 / fan_in)
    
    # Generate uniform distribution within calculated bounds
    return rand(*shape, low=-bound, high=bound, **kwargs)
    ### END YOUR SOLUTION

def kaiming_normal(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    # Calculate shape based on input parameters
    if shape := kwargs.get('shape'):
        fan_in = math.prod(shape[:-1])
    else:
        shape = (fan_in, fan_out)
    
    # Calculate standard deviation using He initialization formula
    std = math.sqrt(2.0) / math.sqrt(fan_in)
    
    # Generate normal distribution with calculated std
    return randn(*shape, mean=0.0, std=std, **kwargs)
    ### END YOUR SOLUTION