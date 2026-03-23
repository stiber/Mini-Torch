# Detailed Specification for `Module` Methods: `Activation`

In the mini-torch framework, the `Activation` class is an abstract base class that inherits from `Module`. It serves as a blueprint for element-wise, non-linear transformations (such as ReLU, Sigmoid, or GELU) that are typically placed between linear layers to allow the network to learn complex, non-linear decision boundaries.

From an object-oriented design perspective, the `Activation` class specializes the `Module` interface by enforcing the rule that activation functions contain **no trainable parameters** (no weights or biases). Therefore, its primary responsibilities are applying a mathematical function during the forward pass and computing the local derivative during the backward pass.

Here is the specification for the `Activation` base class and how it is subclassed:

---

## `__init__()` 

The constructor for an activation layer is drastically simplified compared to a `Linear` layer because it does not need to define input or output dimensions, nor does it initialize weight matrices.

### Method Signature
```python
def __init__(self):
```

### Specification
*   **Superclass Initialization:** Must call `super().__init__()`.
*   **Caching State:** It only needs to establish a single placeholder variable (e.g., `self.x = None`) to cache the input data during the forward pass. This cached input is required to evaluate the derivative of the activation function during backpropagation.

### Example Implementation
```python
def __init__(self):
    super().__init__()
    self.x = None
```

---

## `forward()` 

The forward method applies the specific non-linear mathematical function element-wise across the entire input array.

### Method Signature
```python
def forward(self, x):
```

### Specification
*   **Input (`x`):** A NumPy array of arbitrary shape.
*   **Caching State:** The method must store the input `x` in `self.x` so it can be referenced in the backward pass.
*   **Computation:** Applies the activation function (e.g., thresholding at zero for ReLU or applying the logistic function for Sigmoid).
*   **Returns:** A NumPy array of the exact same shape as the input `x`.

---

## `backward()`

The backward method computes the gradient of the loss with respect to the layer's input by applying the chain rule. 

### Method Signature
```python
def backward(self, grad_output):
```

### Specification
*   **Input (`grad_output`):** A NumPy array representing the incoming gradient from the subsequent layer, possessing the exact same shape as the cached `self.x`.
*   **Computation:** 
    1. Evaluates the local derivative of the specific activation function using the cached `self.x`.
    2. Multiplies this local derivative **element-wise** (Hadamard product) by the incoming `grad_output` to apply the chain rule.
*   **Returns (`grad_input`):** The resulting NumPy array, which is passed backward to the preceding layer. 
*   **No Parameter Updates:** Because there are no weights, it does not calculate or store `self.dW` or `self.db`.

---

## `parameters()` and `grads()` 

Because activation layers have no learnable weights, they must override the `Module` base class methods to return empty lists. This prevents the `Optimizer` from attempting to update non-existent parameters.

### Method Signatures
```python
def parameters(self):
def grads(self):
```

### Specification
*   **Returns:** Both methods simply return an empty Python list: `[]`.

---

## Concrete Subclass Example: `ReLU`

To illustrate how students will use this base class, here is the full implementation of a `ReLU` (Rectified Linear Unit) activation layer. ReLU outputs the input directly if it is positive, and outputs zero if it is negative. Its derivative is 1 for positive inputs and 0 for negative inputs.

```python
class ReLU(Activation):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        self.x = x
        # Element-wise maximum: threshold negative values to 0
        return np.maximum(0, x)
        
    def backward(self, grad_output):
        # Derivative of ReLU is 1 where x > 0, and 0 where x <= 0
        local_derivative = (self.x > 0).astype(float)
        
        # Chain rule: multiply local derivative by incoming gradient element-wise
        grad_input = local_derivative * grad_output
        return grad_input
        
    def parameters(self):
        return []
        
    def grads(self):
        return []
```
