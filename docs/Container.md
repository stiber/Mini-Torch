# Detailed Specification for `Module` Methods: Container

In the mini-torch framework, a container like `Sequential` is a specialized subclass of `Module` that implements the Composite Design Pattern. It holds a list of other `Module` instances (like `Linear` layers or activation functions) and chains them together. Because it inherits from `Module`, it shares the exact same interface as a single layer.

Unlike a single layer, the `Sequential` container does not define its own mathematical weights or perform its own gradient math. Instead, it delegates these responsibilities by iterating through its child modules. 

Here are the detailed specifications for the `Sequential` container methods:

## \_\_init__()

The constructor is responsible for receiving and storing the child layers that make up the sequence.

### Method Signature
```python
def __init__(self, modules):
```

### Specification
*   **Input (`modules`):** A Python `list` (or tuple) containing instances of `Module` subclasses (e.g., `[Linear(2, 10), ReLU(), Linear(10, 2)]`).
*   **State:** The container must store this ordered list in an instance variable (e.g., `self.modules`) so it can be accessed during the forward and backward passes. It does not need to define placeholder variables for caching inputs or gradients, as the individual child modules handle their own caching.

### Example Implementation
```python
def __init__(self, modules):
    self.modules = modules
```

---

##  forward()

The forward method passes the input data sequentially through every layer in the container. 

### Method Signature
```python
def forward(self, x):
```

### Specification
*   **Input (`x`):** A NumPy array of shape `(Batch_Size, Input_Dim)`.
*   **Computation:** The method iterates through the `self.modules` list in standard order. It passes `x` into the first module's `forward()` method, captures the output, and feeds that output directly as the input to the next module's `forward()` method.
*   **Returns:** The final output NumPy array returned by the very last module in the sequence. 

### Example Implementation
```python
def forward(self, x):
    for module in self.modules:
        x = module.forward(x)
    return x
```

---

##  backward()

The backward method implements the chain rule of calculus by propagating the error gradients backwards through the sequence of layers.

### Method Signature
```python
def backward(self, grad_output):
```

### Specification
*   **Input (`grad_output`):** A NumPy array representing the gradient of the loss with respect to the container's final output.
*   **Computation:** The method must iterate through the `self.modules` list in **reverse order**. It passes the incoming `grad_output` into the current module's `backward()` method. The module will return a `grad_input`, which is then used as the `grad_output` for the preceding layer in the reverse iteration.
*   **Returns:** The final `grad_input` returned by the very first module in the sequence (representing the gradient of the loss with respect to the container's original input `x`).

### Example Implementation
```python
def backward(self, grad_output):
    # Iterate through layers from last to first
    for module in reversed(self.modules):
        grad_output = module.backward(grad_output)
    return grad_output
```

---

## parameters() and grads()

To allow the `Optimizer` to update the entire network seamlessly, the container must aggregate the state of all its child modules into flat lists.

### Method Signatures
```python
def parameters(self):
def grads(self):
```

### Specification
*   **`parameters()`:** Iterates through `self.modules` in standard order, calling `.parameters()` on each child. It flattens the returned lists into a single, comprehensive list of all NumPy weight and bias arrays in the entire sequence.
*   **`grads()`:** Iterates through `self.modules` in the exact same order, calling `.grads()` on each child, and flattens the results. 
*   **Crucial Alignment:** Because the container iterates through the same modules in the same order for both methods, the 1-to-1 mapping between parameter arrays and gradient arrays is perfectly preserved for the `Optimizer`.

### Example Implementation
```python
def parameters(self):
    params = []
    for module in self.modules:
        params.extend(module.parameters())
    return params

def grads(self):
    gradients = []
    for module in self.modules:
        gradients.extend(module.grads())
    return gradients
```
