# Detailed Specification for `DataLoader` Methods

In the mini-torch framework, the `DataLoader` class is responsible for orchestrating how data is fed into the neural network during training or evaluation. While the `Dataset` class acts as a simple storage or retrieval mechanism for individual data samples, the `DataLoader` wraps this dataset and handles the complex logic of grouping individual samples into minibatches, shuffling the data to ensure robust training dynamics, and iterating over the dataset one batch at a time. 

From an object-oriented design perspective, the `DataLoader` implements several key principles:
*   **Iterator Pattern (Behavioral):** By implementing Python's magic `__iter__()` method, the `DataLoader` acts as a standard iterable generator. It provides sequential access to batches of data without exposing the underlying mechanics of how the data is indexed, shuffled, or stacked.
*   **Single Responsibility Principle (SRP):** It strictly separates the logic of *data organization* (batching and shuffling) from both the *data retrieval* (handled by `Dataset`) and the *data processing* (handled by the neural network and training loop).
*   **Composition:** Rather than inheriting from a data source, the `DataLoader` accepts a `Dataset` object as a dependency injection in its constructor. This allows developers to use a single `DataLoader` class for any custom `Dataset` they build.

## \_\_init__()

The constructor initializes the data loader by storing a reference to the dataset and setting the hyperparameters that dictate how batches are formed.

### Method Signature
```python
def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False):
```

### Specification
*   **Inputs:**
    *   **`dataset`**: An instance of a `Dataset` subclass that implements `__len__()` and `__getitem__()`.
    *   **`batch_size`**: An `int` specifying how many individual samples should be grouped into a single batch.
    *   **`shuffle`**: A `bool` indicating whether the data should be randomly reordered at the start of every epoch to prevent the network from memorizing sequence patterns.
    *   **`drop_last`**: A `bool` indicating whether to drop the final batch if the total number of samples is not perfectly divisible by the `batch_size`. This is often recommended during training to prevent loss spikes and unstable gradients caused by unusually small batches.
*   **State:** The method must store these arguments as instance variables (e.g., `self.dataset`, `self.batch_size`) to be used during iteration.

### Example Implementation
```python
def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False):
    self.dataset = dataset
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.drop_last = drop_last
```

---

## \_\_iter__()

This method transforms the `DataLoader` object into a Python generator, yielding minibatches of data one at a time for the training loop.

### Method Signature
```python
def __iter__(self):
```

### Specification
*   **Execution Logic:**
    1.  Determine the total number of samples by calling `len(self.dataset)`.
    2.  Generate a list or NumPy array of indices from `0` to `total_samples - 1`.
    3.  If `self.shuffle` is `True`, randomly shuffle these indices.
    4.  Iterate over the indices in chunks of size `self.batch_size`.
    5.  If `self.drop_last` is `True` and the final chunk is smaller than `self.batch_size`, discard it.
*   **Data Aggregation:** For each chunk of indices, call `self.dataset[idx]` to retrieve the individual `(x, y)` samples. 
*   **Tensor Formatting:** Stack the individual `x` arrays and `y` arrays into two unified NumPy arrays. The resulting arrays must follow the framework's standard row-vector notation: `(Batch_Size, Input_Dim)` for inputs and `(Batch_Size, Target_Dim)` for targets.
*   **Yields:** The method must use the `yield` keyword to return the `(batch_x, batch_y)` tuple of NumPy arrays, pausing execution until the next batch is requested by the training loop.

### Example Implementation
```python
def __iter__(self):
    indices = np.arange(len(self.dataset))
    if self.shuffle:
        np.random.shuffle(indices)
        
    for start_idx in range(0, len(indices), self.batch_size):
        batch_indices = indices[start_idx : start_idx + self.batch_size]
        
        # Handle the drop_last condition
        if self.drop_last and len(batch_indices) < self.batch_size:
            break
            
        batch_x, batch_y = [], []
        for idx in batch_indices:
            x, y = self.dataset[idx]
            batch_x.append(x)
            batch_y.append(y)
            
        # Stack individual samples into batch-first NumPy arrays
        yield np.vstack(batch_x), np.vstack(batch_y)
```

---

## \_\_len__()

While not strictly required for the forward or backward pass, implementing the length operator is highly recommended so the training loop can easily calculate progress (e.g., for printing output or showing progress bars).

### Method Signature
```python
def __len__(self):
```

### Specification
*   **Purpose:** Returns the total number of *batches* (not total samples) that the data loader will yield per epoch.
*   **Computation:** Divides the length of the dataset by the batch size. If `drop_last` is `False` and there is a remainder, it rounds up. If `drop_last` is `True`, it rounds down (using integer division).

### Example Implementation
```python
def __len__(self):
    if self.drop_last:
        return len(self.dataset) // self.batch_size
    else:
        return int(np.ceil(len(self.dataset) / self.batch_size))
```
