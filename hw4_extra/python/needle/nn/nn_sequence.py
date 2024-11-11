"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return (1 + ops.exp(-x))**(-1)
        ### END YOUR SOLUTION

class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        # Store configuration
        self.device = device
        self.dtype = dtype
        self.bias = bias
        self.hidden_size = hidden_size
        
        # Calculate initialization bounds
        weight_bound = np.sqrt(1 / hidden_size)
        
        # Initialize weights
        self.W_ih = Parameter(
            init.rand(input_size, hidden_size,
                     low=-weight_bound, high=weight_bound,
                     device=device, dtype=dtype, requires_grad=True)
        )
        self.W_hh = Parameter(
            init.rand(hidden_size, hidden_size,
                     low=-weight_bound, high=weight_bound,
                     device=device, dtype=dtype, requires_grad=True)
        )
        
        # Initialize biases if needed
        if bias:
            self.bias_ih = Parameter(
                init.rand(hidden_size,
                         low=-weight_bound, high=weight_bound,
                         device=device, dtype=dtype, requires_grad=True)
            )
            self.bias_hh = Parameter(
                init.rand(hidden_size,
                         low=-weight_bound, high=weight_bound,
                         device=device, dtype=dtype, requires_grad=True)
            )
        else:
            self.bias_ih = self.bias_hh = None
            
        # Set activation function
        if nonlinearity == "tanh":
            self.nonlinearity = ops.Tanh()
        elif nonlinearity == "relu":
            self.nonlinearity = ops.ReLU()
        else:
            raise ValueError(
                f"Unsupported nonlinearity '{nonlinearity}'. Use 'tanh' or 'relu'."
            )
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        # Get batch size for broadcasting
        batch_size = X.shape[0]
        
        # Initialize hidden state if not provided
        if h is None:
            h = init.zeros(batch_size, self.hidden_size,
                         device=self.device, dtype=self.dtype)
        
        # Compute linear transformations
        input_transform = X @ self.W_ih
        hidden_transform = h @ self.W_hh
        
        # Add biases if present
        if self.bias:
            bias_shape = (1, self.hidden_size)
            broadcast_shape = (batch_size, self.hidden_size)
            
            input_bias = self.bias_ih.reshape(bias_shape).broadcast_to(broadcast_shape)
            hidden_bias = self.bias_hh.reshape(bias_shape).broadcast_to(broadcast_shape)
            
            total_transform = (input_transform + input_bias + 
                             hidden_transform + hidden_bias)
        else:
            total_transform = input_transform + hidden_transform
        
        # Apply nonlinearity
        return self.nonlinearity(total_transform)
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        # Store configuration
        self.device = device
        self.dtype = dtype
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Create RNN cells for each layer
        self.rnn_cells = [
            RNNCell(
                input_size if i == 0 else hidden_size,
                hidden_size,
                bias=bias,
                nonlinearity=nonlinearity,
                device=device,
                dtype=dtype
            )
            for i in range(num_layers)
        ]
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        # Get sequence dimensions
        seq_len, batch_size, _ = X.shape
        
        # Initialize hidden states if not provided
        if h0 is None:
            hidden_states = [
                init.zeros(
                    batch_size, 
                    self.hidden_size,
                    device=self.device,
                    dtype=self.dtype
                )
                for _ in range(self.num_layers)
            ]
        else:
            # Split initial hidden states by layer
            hidden_states = tuple(ops.split(h0, axis=0))
        
        # Split input sequence into time steps
        time_steps = list(ops.split(X, axis=0))
        final_hidden_states = []
        
        # Process through each layer
        for layer_idx, (rnn_cell, layer_hidden) in enumerate(zip(self.rnn_cells, hidden_states)):
            current_hidden = layer_hidden
            
            # Process each time step
            for step_idx, step_input in enumerate(time_steps):
                current_hidden = rnn_cell(step_input, current_hidden)
                if layer_idx < self.num_layers - 1:
                    # Update input for next layer
                    time_steps[step_idx] = current_hidden
            
            final_hidden_states.append(current_hidden)
        
        # Stack outputs and hidden states
        layer_outputs = ops.stack(time_steps, axis=0)
        final_layer_states = ops.stack(final_hidden_states, axis=0)
        
        return layer_outputs, final_layer_states
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        # Store configuration
        self.device = device
        self.dtype = dtype
        self.bias = bias
        self.hidden_size = hidden_size
        
        # Calculate initialization bound
        weight_bound = np.sqrt(1 / hidden_size)
        
        # Initialize weights for input-to-hidden and hidden-to-hidden connections
        self.W_ih = Parameter(
            init.rand(input_size, 4 * hidden_size,
                     low=-weight_bound, high=weight_bound,
                     device=device, dtype=dtype, requires_grad=True)
        )
        self.W_hh = Parameter(
            init.rand(hidden_size, 4 * hidden_size,
                     low=-weight_bound, high=weight_bound,
                     device=device, dtype=dtype, requires_grad=True)
        )
        
        # Initialize biases if needed
        if bias:
            self.bias_ih = Parameter(
                init.rand(4 * hidden_size,
                         low=-weight_bound, high=weight_bound,
                         device=device, dtype=dtype, requires_grad=True)
            )
            self.bias_hh = Parameter(
                init.rand(4 * hidden_size,
                         low=-weight_bound, high=weight_bound,
                         device=device, dtype=dtype, requires_grad=True)
            )
        else:
            self.bias_ih = self.bias_hh = None
            
        # Initialize activation functions
        self.sigmoid = Sigmoid()
        self.tanh = ops.Tanh()
        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        # Get batch size from input
        batch_size = X.shape[0]
        
        # Initialize hidden and cell states if not provided
        if h is None:
            h0 = init.zeros(batch_size, self.hidden_size, 
                          device=self.device, dtype=self.dtype)
            c0 = init.zeros(batch_size, self.hidden_size, 
                          device=self.device, dtype=self.dtype)
        else:
            h0, c0 = h
            
        # Compute gates pre-activation
        if self.bias:
            bias_shape = (1, 4 * self.hidden_size)
            broadcast_shape = (batch_size, 4 * self.hidden_size)
            
            gates_preact = (
                X @ self.W_ih +
                self.bias_ih.reshape(bias_shape).broadcast_to(broadcast_shape) +
                h0 @ self.W_hh +
                self.bias_hh.reshape(bias_shape).broadcast_to(broadcast_shape)
            )
        else:
            gates_preact = X @ self.W_ih + h0 @ self.W_hh
            
        # Split gates and stack them correctly
        gates_split = tuple(ops.split(gates_preact, axis=1))
        gates = [
            ops.stack(gates_split[i * self.hidden_size : (i + 1) * self.hidden_size], axis=1)
            for i in range(4)
        ]
        
        # Apply activations to get input, forget, cell, and output gates
        input_gate, forget_gate, cell_gate, output_gate = gates
        input_gate = self.sigmoid(input_gate)
        forget_gate = self.sigmoid(forget_gate)
        cell_gate = self.tanh(cell_gate)
        output_gate = self.sigmoid(output_gate)
        
        # Compute new cell and hidden states
        cell_state = forget_gate * c0 + input_gate * cell_gate
        hidden_state = output_gate * self.tanh(cell_state)
        
        return hidden_state, cell_state
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        # Store configuration
        self.device = device
        self.dtype = dtype
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Create LSTM cells for each layer
        self.lstm_cells = [
            LSTMCell(
                input_size if i == 0 else hidden_size,
                hidden_size,
                bias=bias,
                device=device,
                dtype=dtype
            )
            for i in range(num_layers)
        ]
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        # Get sequence dimensions
        seq_len, batch_size, _ = X.shape
        
        # Initialize hidden and cell states if not provided
        if h is None:
            hidden_states = [
                init.zeros(batch_size, self.hidden_size,
                          device=self.device, dtype=self.dtype)
                for _ in range(self.num_layers)
            ]
            cell_states = [
                init.zeros(batch_size, self.hidden_size,
                          device=self.device, dtype=self.dtype)
                for _ in range(self.num_layers)
            ]
        else:
            # Split provided states by layer
            hidden_states = tuple(ops.split(h[0], axis=0))
            cell_states = tuple(ops.split(h[1], axis=0))
        
        # Process sequence through LSTM layers
        final_hidden_states = []
        final_cell_states = []
        time_steps = list(ops.split(X, axis=0))
        
        # Process each layer
        for layer_idx in range(self.num_layers):
            current_hidden = hidden_states[layer_idx]
            current_cell = cell_states[layer_idx]
            lstm_cell = self.lstm_cells[layer_idx]
            
            # Process sequence through current layer
            for step_idx, step_input in enumerate(time_steps):
                current_hidden, current_cell = lstm_cell(
                    step_input,
                    (current_hidden, current_cell)
                )
                time_steps[step_idx] = current_hidden
            
            # Store final states
            final_hidden_states.append(current_hidden)
            final_cell_states.append(current_cell)
        
        # Stack outputs and states
        outputs = ops.stack(time_steps, axis=0)
        final_states = (
            ops.stack(final_hidden_states, axis=0),
            ops.stack(final_cell_states, axis=0)
        )
        
        return outputs, final_states
        ### END YOUR SOLUTION

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        # Store configuration
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        # Initialize embedding weights from normal distribution
        self.weight = Parameter(
            init.randn(
                num_embeddings,
                embedding_dim,
                mean=0.0,
                std=1.0,
                device=device,
                dtype=dtype,
                requires_grad=True
            )
        )
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        # Get sequence dimensions
        sequence_length, batch_size = x.shape
        
        # Convert indices to one-hot vectors
        one_hot_vectors = init.one_hot(
            self.num_embeddings,
            x,
            device=x.device,
            dtype=x.dtype
        )
        
        # Multiply with embedding weights
        flat_embeddings = (one_hot_vectors.reshape((sequence_length * batch_size, self.num_embeddings)) 
                         @ self.weight)
        
        # Reshape to sequence format
        embeddings = flat_embeddings.reshape((sequence_length, batch_size, self.embedding_dim))
        
        return embeddings
        ### END YOUR SOLUTION