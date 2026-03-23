# Detailed Specification for `Optimizer` Methods

In the mini-torch framework, the `Optimizer` class maintains references to each module's (layer's) parameters and their gradient information. It implements the **Single Responsibility Principle (SRP)** in OO design by taking responsibility for parameter updates.

###  \_\_init__()`

The constructor for the `Optimizer` is responsible for receiving the components it needs to update and storing the hyperparameters for the optimization algorithm. 

### Method Signature
```python
def __init__(self, modules, lr=0.01):
```

### Specification
*   **Inputs:** 
    *   **`modules`**: A Python `list` containing instances of the `Module` subclasses that make up the neural network. *(Note: In the provided mini-torch spec, the optimizer takes a list of modules directly rather than a flattened list of parameters as seen in native PyTorch.)*
    *   **`lr`**: A `float` representing the **learning rate**, which controls the size of the steps taken during weight updates.
*   **State:** The method must store the provided modules and learning rate in instance variables (e.g., `self.modules` and `self.lr`) so they can be accessed during the training loop.

### Example Implementation (for an SGD Optimizer)
```python
def __init__(self, modules, lr=0.01):
    self.modules = modules
    self.lr = lr
```

---

##  step() 

The `step()` method is the core engine of the optimizer. It is called once per batch to apply the gradients calculated during the backward pass and mathematically update the network's weights.

### Method Signature
```python
def step(self):
```

### Specification
*   **Execution:** The method **iterates through every module** stored in `self.modules`.
*   **State Retrieval:** For each module, it calls the `parameters()` and `grads()` methods to retrieve the lists of trainable parameter arrays and their corresponding gradient arrays.
*   **Parameter Update:** It iterates through these aligned lists and **modifies each parameter in-place**. For a standard Stochastic Gradient Descent (SGD) algorithm, this means subtracting the gradient multiplied by the learning rate from the current parameter value (`p = p - lr * grad`).

### Example Implementation (for an SGD Optimizer)
```python
def step(self):
    for module in self.modules:
        params = module.parameters()
        grads = module.grads()
        
        # Update each parameter: p = p - lr * grad
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]
```

---

##  zero_grad()

The `zero_grad()` method clears the gradient caches from the previous training step. 

### Method Signature
```python
def zero_grad(self):
```

### Specification
*   **Purpose:** In frameworks relying on manual gradient calculation or backpropagation, gradients are typically accumulated (added together) by default. This method explicitly **resets gradients to zero** before starting a new forward/backward pass on a new batch. 
*   **Importance:** If gradients are not zeroed out in each update round, they will accumulate across batches, leading to incorrect weight updates and broken training dynamics.
*   **Execution:** The method iterates through `self.modules` and triggers the gradient reset mechanism for each layer (such as calling a helper `zero_grad()` function on the `Module` itself, or explicitly setting the `dW` and `db` arrays to zero).

###  Example Implementation
```python
def zero_grad(self):
    for module in self.modules:
        if hasattr(module, 'zero_grad'):
            module.zero_grad()
```
