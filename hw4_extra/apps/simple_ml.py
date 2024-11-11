"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl

import needle.nn as nn
from apps.models import *
import time
device = ndl.cpu()

def parse_mnist(image_filesname, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    # MNIST constants
    IMAGE_HEADER_BYTES = 16  # Size of the header for image file
    LABEL_HEADER_BYTES = 8   # Size of the header for label file
    PIXEL_DEPTH = 255.0      # Maximum pixel value for normalization
    IMAGE_SIZE = 784         # Flattened image size (28x28)
    
    # Load and process images
    try:
        with gzip.open(image_filesname, 'rb') as image_file:
            # Read raw image data and skip header bytes
            raw_image_data = image_file.read()
            image_array = np.frombuffer(
                raw_image_data, 
                dtype=np.uint8, 
                offset=IMAGE_HEADER_BYTES
            )
            
            # Normalize and reshape images
            normalized_images = (image_array.astype(np.float32) / PIXEL_DEPTH)
            flattened_images = np.reshape(normalized_images, (-1, IMAGE_SIZE))
            
        # Load and process labels
        with gzip.open(label_filename, "rb") as label_file:
            # Read raw label data and skip header bytes
            raw_label_data = label_file.read()
            label_array = np.frombuffer(
                raw_label_data, 
                dtype=np.uint8, 
                offset=LABEL_HEADER_BYTES
            )
            
        return flattened_images, label_array
        
    except FileNotFoundError as e:
        raise FileNotFoundError(f"MNIST data files not found: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Error processing MNIST data: {str(e)}")
    ### END YOUR SOLUTION


def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    # Get batch size for averaging
    batch_size = Z.shape[0]
    log_sum_exp = ndl.log(ndl.summation(ndl.exp(Z), axes=1))
    true_class_logits = ndl.summation(y_one_hot * Z, axes=1)
    total_loss = ndl.summation(log_sum_exp - true_class_logits)
    return total_loss / batch_size
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    num_examples = X.shape[0]
    num_batches = num_examples // batch
    num_classes = W2.shape[1]
    
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch
        batch_end = (batch_idx + 1) * batch
        batch_inputs = ndl.Tensor(X[batch_start:batch_end], dtype="float32")
        batch_labels = y[batch_start:batch_end]
        
        # Forward pass
        # First layer: input -> hidden
        hidden_activations = ndl.matmul(batch_inputs, W1)
        hidden_outputs = ndl.relu(hidden_activations)
        
        # Second layer: hidden -> output
        logits = ndl.matmul(hidden_outputs, W2)
        
        # Convert labels to one-hot encoding
        one_hot_labels = np.zeros((batch, num_classes))
        one_hot_labels[np.arange(batch), batch_labels] = 1
        one_hot_labels = ndl.Tensor(one_hot_labels, dtype="float32")
        
        # Compute loss
        batch_loss = softmax_loss(logits, one_hot_labels)
        
        # Backward pass
        batch_loss.backward()
        
        # Update weights using gradient descent
        # Note: realize_cached_data() is used to get the actual numpy array
        W1 = ndl.Tensor(
            W1.realize_cached_data() - lr * W1.grad.realize_cached_data()
        )
        W2 = ndl.Tensor(
            W2.realize_cached_data() - lr * W2.grad.realize_cached_data()
        )
        
    return W1, W2
    ### END YOUR SOLUTION

### CIFAR-10 training ###
def epoch_general_cifar10(dataloader, model, loss_fn=nn.SoftmaxLoss(), opt=None):
    """
    Iterates over the dataloader. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    total_correct = 0
    total_loss = 0
    
    # Set model mode based on whether we're training or evaluating
    model.train() if opt is not None else model.eval()
    
    # Process each batch
    for batch_inputs, batch_labels in dataloader:
        # Convert numpy arrays to tensors
        batch_inputs = ndl.Tensor(batch_inputs, device=device)
        batch_labels = ndl.Tensor(batch_labels, device=device)
        
        # Reset gradients if in training mode
        if opt is not None:
            opt.reset_grad()
            
        # Forward pass
        predictions = model(batch_inputs)
        batch_loss = loss_fn()(predictions, batch_labels)
        
        # Training step if optimizer is provided
        if opt is not None:
            batch_loss.backward()
            opt.step()
        
        # Calculate metrics
        # Convert predictions to numpy for accuracy calculation
        pred_labels = np.argmax(predictions.numpy(), axis=1)
        true_labels = batch_labels.numpy()
        
        # Update running metrics
        batch_correct = np.sum(pred_labels == true_labels)
        total_correct += batch_correct
        
        # Accumulate loss (handle both training and eval cases)
        if opt is not None:
            batch_loss_value = batch_loss.numpy()
        else:
            batch_loss_value = batch_loss.data.numpy()
            
        total_loss += batch_loss_value * batch_labels.shape[0]
    
    # Calculate final metrics
    num_samples = len(dataloader.dataset)
    average_accuracy = total_correct / num_samples
    average_loss = total_loss / num_samples
    
    return average_accuracy, average_loss
    ### END YOUR SOLUTION


def train_cifar10(model, dataloader, n_epochs=1, optimizer=ndl.optim.Adam,
          lr=0.001, weight_decay=0.001, loss_fn=nn.SoftmaxLoss):
    """
    Performs {n_epochs} epochs of training.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    # Initialize optimizer with model parameters
    optimizer_instance = optimizer(
        params=model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    
    # Track best metrics
    best_accuracy = 0.0
    best_loss = float('inf')
    
    # Training loop
    for epoch in range(n_epochs):
        # Perform one epoch of training
        epoch_accuracy, epoch_loss = epoch_general_cifar10(
            dataloader=dataloader,
            model=model,
            loss_fn=loss_fn,
            opt=optimizer_instance
        )
        
        # Update best metrics
        best_accuracy = max(best_accuracy, epoch_accuracy)
        best_loss = min(best_loss, epoch_loss)
        
        # Log progress
        print(
            f"Epoch [{epoch+1}/{n_epochs}] | "
            f"Accuracy: {epoch_accuracy:.4f} | "
            f"Loss: {epoch_loss:.4f} | "
            f"Best Accuracy: {best_accuracy:.4f} | "
            f"Best Loss: {best_loss:.4f}"
        )
    
    # Return final epoch metrics
    return epoch_accuracy, epoch_loss
    ### END YOUR SOLUTION


def evaluate_cifar10(model, dataloader, loss_fn=nn.SoftmaxLoss):
    """
    Computes the test accuracy and loss of the model.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    accuracy, loss = epoch_general_cifar10(
        dataloader=dataloader,
        model=model,
        loss_fn=loss_fn,
        opt=None  # None indicates evaluation mode
    )
    
    # Print detailed evaluation metrics
    print("=" * 50)
    print("Evaluation Results:")
    print("-" * 50)
    print(f"Classification Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Average Loss: {loss:.6f}")
    print("=" * 50)
    
    return accuracy, loss
    ### END YOUR SOLUTION


### PTB training ###
def epoch_general_ptb(data, model, seq_len=40, loss_fn=nn.SoftmaxLoss(), opt=None,
        clip=None, device=None, dtype="float32"):
    """
    Iterates over the data. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        data: data of shape (nbatch, batch_size) given from batchify function
        model: LanguageModel instance
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    # Set model mode
    model.train() if opt is not None else model.eval()
    
    # Initialize metrics and get data dimensions
    batch_metrics = {
        'accuracies': [],
        'losses': []
    }
    num_batches, batch_size = data.shape
    hidden_state = None
    
    # Process data in sequences
    for batch_start in range(0, num_batches - 1, seq_len):
        # Get current sequence batch
        inputs, targets = ndl.data.get_batch(
            data=data,
            i=batch_start,
            bptt=seq_len,
            device=device,
            dtype=dtype
        )
        
        # Forward pass
        predictions, hidden_state = model()(inputs, hidden_state)
        batch_loss = loss_fn()(predictions, targets)
        
        # Training step if in training mode
        if opt is not None:
            # Reset gradients and compute backward pass
            opt.reset_grad()
            batch_loss.backward()
            
            # Gradient clipping (commented out as per original)
            # if clip is not None:
            #     ndl.nn.utils.clip_grad_norm(model.parameters(), clip)
            
            # Update weights
            opt.step()
        
        # Detach hidden state for truncated BPTT
        if isinstance(hidden_state, tuple):
            hidden_state = tuple(h.detach() for h in hidden_state)
        else:
            hidden_state = hidden_state.detach()
        
        # Calculate and store batch metrics
        batch_predictions = predictions.numpy().argmax(axis=1)
        batch_targets = targets.numpy()
        batch_accuracy = np.sum(batch_predictions == batch_targets) / targets.shape[0]
        
        batch_metrics['accuracies'].append(batch_accuracy)
        batch_metrics['losses'].append(batch_loss.numpy())
        
        # Clean up tensors to manage memory
        del inputs, targets, predictions, batch_loss
    
    # Calculate average metrics
    average_accuracy = np.mean(batch_metrics['accuracies'])
    average_loss = np.mean(batch_metrics['losses'])
    
    return average_accuracy, average_loss
    ### END YOUR SOLUTION


def train_ptb(model, data, seq_len=40, n_epochs=1, optimizer=ndl.optim.SGD,
          lr=4.0, weight_decay=0.0, loss_fn=nn.SoftmaxLoss, clip=None,
          device=None, dtype="float32"):
    """
    Performs {n_epochs} epochs of training.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    # Initialize optimizer
    optimizer_instance = optimizer(
        params=model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    
    # Track best metrics
    best_metrics = {
        'accuracy': 0.0,
        'loss': float('inf'),
        'epoch': 0
    }
    
    # Training loop
    for epoch in range(n_epochs):
        # Train one epoch
        epoch_accuracy, epoch_loss = epoch_general_ptb(
            data=data,
            model=model,
            seq_len=seq_len,
            loss_fn=loss_fn,
            opt=optimizer_instance,
            clip=clip,
            device=device,
            dtype=dtype
        )
        
        # Update best metrics
        if epoch_accuracy > best_metrics['accuracy']:
            best_metrics.update({
                'accuracy': epoch_accuracy,
                'loss': epoch_loss,
                'epoch': epoch + 1
            })
        
        # Log progress
        print(f"Epoch [{epoch+1}/{n_epochs}]")
        print(f"  ├─ Accuracy: {epoch_accuracy:.4f}")
        print(f"  ├─ Loss: {epoch_loss:.4f}")
        print(f"  └─ Best Accuracy: {best_metrics['accuracy']:.4f} (Epoch {best_metrics['epoch']})")
        
    return epoch_accuracy, epoch_loss
    ### END YOUR SOLUTION

def evaluate_ptb(model, data, seq_len=40, loss_fn=nn.SoftmaxLoss,
        device=None, dtype="float32"):
    """
    Computes the test accuracy and loss of the model.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    # Perform evaluation
    accuracy, loss = epoch_general_ptb(
        data=data,
        model=model,
        seq_len=seq_len,
        loss_fn=loss_fn,
        opt=None,  # None indicates evaluation mode
        device=device,
        dtype=dtype
    )
    
    # Print detailed evaluation results
    print("=" * 50)
    print("PTB Evaluation Results:")
    print("-" * 50)
    print(f"Sequence Length: {seq_len}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Loss: {loss:.6f}")
    print(f"Perplexity: {np.exp(loss):.2f}")
    print("=" * 50)
    
    return accuracy, loss
    ### END YOUR SOLUTION

### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
