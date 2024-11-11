"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for param in self.params:
            # Compute gradient with weight decay
            grad = (param.grad.data + self.weight_decay * param.data)
            
            # Update momentum buffer
            self.u[param] = ndl.Tensor(
                self.momentum * self.u.get(param, 0) + 
                (1 - self.momentum) * grad,
                dtype=param.dtype
            )
            
            # Update parameter
            param.data -= self.lr * self.u[param]
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        # Compute total norm of all gradients
        grad_norms = [np.linalg.norm(p.grad.detach().numpy()).reshape((1,)) 
                     for p in self.params]
        total_norm = np.linalg.norm(np.array(grad_norms))
        
        # Compute clipping coefficient
        clip_coef = max_norm / (total_norm + 1e-6)
        clip_coef_clamped = min((np.asscalar(clip_coef), 1.0))
        
        # Apply clipping to all gradients
        for p in self.params:
            p.grad = p.grad.detach() * clip_coef_clamped
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        
        for param in self.params:
            # Compute gradient with weight decay
            grad = param.grad.data + self.weight_decay * param.data
            
            # Update biased first moment estimate
            self.m[param] = (self.beta1 * self.m.get(param, 0) + 
                           (1 - self.beta1) * grad)
            
            # Update biased second moment estimate
            self.v[param] = (self.beta2 * self.v.get(param, 0) + 
                           (1 - self.beta2) * (grad ** 2))
            
            # Compute bias-corrected moment estimates
            m_hat = self.m[param] / (1 - self.beta1 ** self.t)
            v_hat = self.v[param] / (1 - self.beta2 ** self.t)
            
            # Compute update
            update = ndl.Tensor(
                self.lr * m_hat.data / (v_hat.data ** 0.5 + self.eps),
                dtype=param.dtype
            )
            
            # Apply update to parameter
            param.data -= update.data
        ### END YOUR SOLUTION
