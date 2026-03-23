# Detailed Specification for `Module` Methods: Single-Layer

## \_\_init__()


For a single-layer subclass of `Module` (such as a `Linear` or Dense layer) in the mini-torch framework, the `__init__()` method is responsible for setting up the layer's dimensions, initializing its trainable parameters using NumPy, and preparing placeholder variables to cache data for manual gradient calculations.

Here is the specification for the `__init__()` method:

### Method Signature
The method should accept the dimensions required to build the weight matrix and bias vector.
```python
def __init__(self, input_dim, output_dim):
```

### Trainable Parameter Initialization
The layer must define its internal weights and biases as NumPy arrays. Because mini-torch uses row-vector (batch-first) notation, the dimensions must be strictly ordered to support `x @ W + b`.
*   **Weight Matrix (`self.W`):** Should be initialized as a NumPy array of shape `(input_dim, output_dim)`. It is highly recommended to use *He* Initialization (generating random values scaled by the square root of 2 divided by the input dimension) to prevent vanishing or exploding gradients.
*   **Bias Vector (`self.b`):** Should be initialized as a NumPy array of zeros with shape `(1, output_dim)`. This shape ensures it broadcasts correctly across the batch dimension during the forward pass.

### Caching for the Backward Pass
Because the mini-torch framework requires manual backpropagation, the layer must cache inputs and gradients. The `__init__()` method must define these state variables and initialize them to `None`.
*   **`self.x = None`**: A placeholder to store the input data during the `forward` pass, which is required later to calculate the weight gradients (`dW = x.T @ grad_output`).
*   **`self.dW = None`**: A placeholder to store the calculated gradient of the loss with respect to the weights.
*   **`self.db = None`**: A placeholder to store the calculated gradient of the loss with respect to the biases.

### Example Implementation
Based on the framework's core philosophy, the complete `__init__()` method for a `Linear` layer looks like this:

```python
def __init__(self, input_dim, output_dim):
    # He Initialization for weights
    self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
    # Zeros for biases
    self.b = np.zeros((1, output_dim))
    
    # Cache for backward pass
    self.x = None
    self.dW = None
    self.db = None
```

## forward()

The `forward()` method defines the computation performed at every call of the layer. It takes the input data, applies the layer's mathematical operations, caches necessary data for the backpropagation step, and returns the output.

### Method Signature
```python
def forward(self, x):
```

### Specification
*   **Input (`x`):** A NumPy array of shape `(Batch_Size, Input_Dim)`. Because the framework uses batch-first, row-vector notation, each row represents a separate example in the batch.
*   **Caching State:** Before returning the result, the method must save the input `x` to the instance variable `self.x` (e.g., `self.x = x`). This cached input is strictly required during the backward pass to calculate the gradients for the weights.
*   **Computation:** The layer performs the linear transformation using the `@` operator for matrix multiplication: `x @ self.W + self.b`.
*   **Returns:** A NumPy array representing the layer's output, with shape `(Batch_Size, Output_Dim)`.

### Example Implementation
```python
def forward(self, x):
    self.x = x  # Cache for backward pass
    return x @ self.W + self.b
```


## backward()

The `backward()` method is responsible for manual gradient calculation using the chain rule of calculus. It computes how much the loss function changes with respect to the layer's parameters (to update the weights) and with respect to the layer's inputs (to continue the chain rule backwards to previous layers).

### Method Signature
```python
def backward(self, grad_output):
```

### Specification
*   **Input (`grad_output`):** A NumPy array representing the gradient of the loss with respect to this specific layer's output. It will have the shape `(Batch_Size, Output_Dim)`. This is passed in from the subsequent layer (or from the `Loss` module if this is the final layer).
*   **Parameter Gradient Calculation (State Update):** The method must calculate the gradients for the internal parameters and store them in the placeholder variables defined in `__init__()`:
    *   **Weight Gradients (`self.dW`):** Calculated using the dot product of the transposed cached input and the incoming gradient: `self.dW = self.x.T @ grad_output`.
    *   **Bias Gradients (`self.db`):** Calculated by summing the `grad_output` across the batch dimension (axis 0). You must use `keepdims=True` to maintain the `(1, Output_Dim)` shape so it aligns with `self.b`: `self.db = np.sum(grad_output, axis=0, keepdims=True)`.
*   **Returns (`grad_input`):** The method must calculate and return the gradient of the loss with respect to the layer's *inputs* so that the preceding layer can use it as its `grad_output`. This is calculated as: `grad_input = grad_output @ self.W.T`.

### Example Implementation
```python
def backward(self, grad_output):
    # 1. Calculate and store gradients w.r.t. parameters
    self.dW = self.x.T @ grad_output
    self.db = np.sum(grad_output, axis=0, keepdims=True)
    
    # 2. Calculate and return gradient w.r.t. the layer's input
    grad_input = grad_output @ self.W.T
    return grad_input
```
---

## Optimizer Interface

The `parameters()` and `grads()`  methods work in tandem to expose the layer's internal state to the `Optimizer` class, allowing the optimizer to update the weights without needing to know the specific details of the layer's architecture.

## parameters()

The `parameters()` method provides access to the layer's trainable variables. 

### Method Signature
```python
def parameters(self):
```

### Specification
*   **Purpose:** It gathers all the learnable weights and biases defined during `__init__()` that need to be updated during the training loop.
*   **Returns:** A standard Python `list` containing the NumPy arrays representing the layer's parameters.
*   **Crucial Alignment:** The order of the parameters in this list is completely up to the developer, but it **must exactly match** the order of the gradients returned by the `grads()` method. If the weight matrix is first in this list, its corresponding gradient must be first in the `grads()` list.

### Example Implementation
```python
def parameters(self):
    # Returns the weight matrix and bias vector
    return [self.W, self.b]
```


##  grads() 

The `grads()` method provides access to the gradients calculated during the `backward()` pass.

### Method Signature
```python
def grads(self):
```

### Specification
*   **Purpose:** It gathers the gradient placeholders (which were populated during the manual backpropagation step) so the optimizer knows how much to adjust each parameter.
*   **Returns:** A standard Python `list` containing the NumPy arrays representing the gradients of the loss with respect to the layer's parameters.
*   **Crucial Alignment:** As noted above, the length of this list and the order of its elements must perfectly map 1-to-1 with the list returned by `parameters()`.

### Example Implementation
```python
def grads(self):
    # Returns the gradients cached during the backward() pass
    return [self.dW, self.db]
```

### How They Interact with the Optimizer
To understand why this spec is written this way, it is helpful to look at how the `Optimizer` relies on them. When `optimizer.step()` is called, it iterates through both lists simultaneously to apply the update rule (like Stochastic Gradient Descent):

```python
# Inside the Optimizer's step() method:
params = module.parameters()
grads = module.grads()

# The strict 1-to-1 alignment allows this simple loop:
for i in range(len(params)):
    params[i] -= self.lr * grads[i]
```
