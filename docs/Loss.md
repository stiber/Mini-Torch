# Detailed Specification for `Loss` Methods

In the mini-torch framework, the `Loss` class acts as an abstract base class that quantifies the difference between the neural network's predictions and the actual target values. It serves a dual purpose: providing a scalar metric to evaluate the model's performance during training and validation, and initiating the backpropagation process by calculating the foundational error gradient.

From an object-oriented design perspective, the `Loss` class implements the **Strategy Pattern**. By encapsulating the specific error calculation algorithms (such as Mean Squared Error or Cross-Entropy) inside interchangeable subclasses, students can swap out the loss function without needing to alter the core model architecture or the training loop.

Because the mini-torch framework relies on manual gradient calculations rather than an autograd engine, the `Loss` class must cache the network's predictions and the true targets during the forward pass so they can be used to compute the derivative during the backward pass.

## \_\_init__()

The constructor simply initializes the state variables used for caching data between the forward and backward passes.

**Method Signature:**
```python
def __init__(self):
```

**Specification:**
*   **State:** The method must define instance variables to temporarily store the outputs of the network and the target labels (e.g., `self.predictions = None` and `self.targets = None`).

**Example Implementation:**
```python
def __init__(self):
    self.predictions = None
    self.targets = None
```

---

##  forward()

The forward method calculates the scalar error metric based on the provided predictions and targets.

**Method Signature:**
```python
def forward(self, predictions, targets):
```

**Specification:**
*   **Inputs:**
    *   **`predictions`**: A NumPy array of shape `(Batch_Size, Output_Dim)` representing the final output of the neural network (often called logits before a softmax, or raw continuous outputs).
    *   **`targets`**: A NumPy array representing the desired true values or class labels. 
*   **Caching State:** The method must store both `predictions` and `targets` in the instance variables defined in `__init__()` for later use in gradient computation.
*   **Computation:** Applies the specific mathematical formula for the chosen loss metric (e.g., averaging the squared differences or computing the negative log probability).
*   **Returns:** A single Python `float` representing the average loss across the batch.

**Example Implementation (for Mean Squared Error):**
```python
def forward(self, predictions, targets):
    self.predictions = predictions
    self.targets = targets
    
    # Calculate Mean Squared Error
    return np.mean((self.predictions - self.targets) ** 2)
```

---

## backward()

The backward method calculates the derivative of the loss function with respect to the network's predictions, providing the initial `grad_output` that kicks off the chain rule for the rest of the network.

**Method Signature:**
```python
def backward(self):
```

**Specification:**
*   **Inputs:** None (it relies entirely on the cached `self.predictions` and `self.targets`).
*   **Computation:** Computes the mathematical derivative of the specific loss function with respect to the `predictions` array. 
*   **Returns (`grad_output`):** A NumPy array of the exact same shape as `predictions`: `(Batch_Size, Output_Dim)`. This array is returned to the training loop, where it is passed directly into the final `Module`'s `backward(grad_output)` method.

**Example Implementation (for Mean Squared Error):**
```python
def backward(self):
    batch_size = self.predictions.shape
    output_dim = self.predictions.shape
    
    # Derivative of MSE: (2 / N) * (predictions - targets)
    # where N is the total number of elements (batch_size * output_dim)
    grad_output = (2.0 / (batch_size * output_dim)) * (self.predictions - self.targets)
    
    return grad_output
```

## Subclass Note: Cross-Entropy Loss

When implementing classification networks, students will subclass `Loss` to create a `CrossEntropyLoss` module. 

In modern deep learning frameworks like PyTorch, it is standard practice for efficiency and numerical stability to combine the Softmax activation function and the Negative Log-Likelihood loss into a single class rather than computing them as separate layers.

If students implement `CrossEntropyLoss`, the specification should direct them to:
1.  **Forward Pass:** Apply the softmax function to the incoming `predictions` (logits) internally to get probabilities, then calculate the negative average log probability using the `targets`.
2.  **Backward Pass:** Rely on the elegantly simplified gradient of Softmax combined with Cross-Entropy. If targets are one-hot encoded, the gradient simplifies to just: `probabilities - targets`. 

**Example Skeleton for `CrossEntropyLoss`:**
```python
class CrossEntropyLoss(Loss):
    def forward(self, predictions, targets):
        # 1. Apply softmax to predictions for numerical stability
        # 2. Cache probabilities and targets
        # 3. Calculate and return the negative average log probability
        pass
        
    def backward(self):
        # Return the simplified gradient: (probabilities - targets) / batch_size
        pass
```
