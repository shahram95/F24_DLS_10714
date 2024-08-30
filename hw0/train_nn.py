import sys
sys.path.append("src/")

# Reload the simple_ml module which has been cached from the earlier experiment
import importlib
import simple_ml
importlib.reload(simple_ml)

sys.path.append("src/")
from simple_ml import train_nn, parse_mnist

X_tr, y_tr = parse_mnist("data/train-images-idx3-ubyte.gz", 
                         "data/train-labels-idx1-ubyte.gz")
X_te, y_te = parse_mnist("data/t10k-images-idx3-ubyte.gz",
                         "data/t10k-labels-idx1-ubyte.gz")
train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=400, epochs=20, lr=0.2)