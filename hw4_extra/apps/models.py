import sys
sys.path.append('./python')
from needle.autograd import Tensor
import needle as ndl
import needle.nn as nn
import math
import numpy as np
np.random.seed(0)

def create_conv_block(input_channels, output_channels, kernel_size, stride, device=None, dtype="float32"):
    """Creates a convolutional block with batch normalization and ReLU activation."""
    return nn.Sequential(
        nn.Conv(input_channels, output_channels, kernel_size=kernel_size, stride=stride, bias=True, device=device, dtype=dtype),
        nn.BatchNorm2d(output_channels, device=device, dtype=dtype),
        nn.ReLU()
    )

class ResidualBlock(ndl.nn.Module):
    """A residual block that performs skip connections with two convolutional layers."""
    def __init__(self, input_channels, output_channels, kernel_size=3, stride=1, device=None, dtype="float32"):
        super().__init__()
        self.main_path_conv1 = create_conv_block(input_channels, output_channels, kernel_size, stride, device=device, dtype=dtype)
        self.main_path_conv2 = create_conv_block(output_channels, output_channels, kernel_size, stride, device=device, dtype=dtype)
        
    def forward(self, input_tensor):
        residual_connection = input_tensor
        main_path_output = self.main_path_conv1(input_tensor)
        main_path_output = self.main_path_conv2(main_path_output)
        return main_path_output + residual_connection
    
class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION ###
        self.entry_layer = create_conv_block(3, 16, kernel_size=7, stride=4, device=device, dtype=dtype)
        self.downsampling_layer1 = create_conv_block(16, 32, kernel_size=3, stride=2, device=device, dtype=dtype)
        self.residual_block1 = ResidualBlock(32, 32, kernel_size=3, stride=1, device=device, dtype=dtype)
        self.downsampling_layer2 = create_conv_block(32, 64, kernel_size=3, stride=2, device=device, dtype=dtype)
        self.downsampling_layer3 = create_conv_block(64, 128, kernel_size=3, stride=2, device=device, dtype=dtype)
        self.residual_block2 = ResidualBlock(128, 128, kernel_size=3, stride=1, device=device, dtype=dtype)
        self.feature_layer = nn.Linear(128, 128, device=device, dtype=dtype)
        self.output_layer = nn.Linear(128, 10, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x):
        ### BEGIN YOUR SOLUTION
        features = self.entry_layer(x)
        features = self.downsampling_layer1(features)
        features = self.residual_block1(features)
        features = self.downsampling_layer2(features)
        features = self.downsampling_layer3(features)
        features = self.residual_block2(features)
        features = nn.Flatten()(features)
        features = self.feature_layer(features)
        features = ndl.ops.relu(features)
        logits = self.output_layer(features)
        return logits
        ### END YOUR SOLUTION


class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', seq_len=40, device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        ### BEGIN YOUR SOLUTION
        # Store model dimensions
        self.embedding_dim = embedding_size
        self.vocab_size = output_size
        self.hidden_dim = hidden_size
        
        # Word embedding layer
        self.word_embedding = nn.Embedding(
            num_embeddings=self.vocab_size, 
            embedding_dim=self.embedding_dim, 
            device=device, 
            dtype=dtype
        )
        
        # Sequence model selection
        if seq_model == 'rnn':
            self.sequence_model = nn.RNN(
                input_size=self.embedding_dim,
                hidden_size=self.hidden_dim,
                num_layers=num_layers,
                device=device,
                dtype=dtype
            )
        elif seq_model == 'lstm':
            self.sequence_model = nn.LSTM(
                input_size=self.embedding_dim,
                hidden_size=self.hidden_dim,
                num_layers=num_layers,
                device=device,
                dtype=dtype
            )
        elif seq_model == 'transformer':
            self.sequence_model = nn.Transformer(
                input_size=self.embedding_dim,
                hidden_size=self.hidden_dim,
                num_layers=num_layers,
                device=device,
                dtype=dtype,
                sequence_len=seq_len
            )
        else:
            raise ValueError('Supported sequence models are: rnn, lstm, transformer')
        
        # Output projection layer
        self.output_projection = nn.Linear(
            in_features=self.hidden_dim,
            out_features=self.vocab_size,
            device=device,
            dtype=dtype
        )
        ### END YOUR SOLUTION

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        ### BEGIN YOUR SOLUTION
        # Get sequence dimensions
        input_sequence = x
        hidden_state = h
        sequence_length, batch_size = input_sequence.shape
        
        # Embed input tokens
        embedded_sequence = self.word_embedding(input_sequence)
        
        # Process through sequence model
        sequence_outputs, final_hidden_state = self.sequence_model(embedded_sequence, hidden_state)
        
        # Reshape for output projection
        flattened_outputs = sequence_outputs.reshape((sequence_length * batch_size, self.hidden_dim))
        
        # Project to vocabulary size
        output_logits = self.output_projection(flattened_outputs)
        
        return output_logits, final_hidden_state
        ### END YOUR SOLUTION


if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = ndl.data.DataLoader(cifar10_train_dataset, 128, ndl.cpu(), dtype="float32")
    print(cifar10_train_dataset[1][0].shape)
