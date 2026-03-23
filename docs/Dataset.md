# Detailed Specification for `Dataset` Methods

In the mini-torch framework, the `Dataset` class serves as an abstract base class that defines how individual data records and their corresponding labels are loaded and accessed. While the `DataLoader` class is responsible for the complex orchestration of batching, shuffling, and multi-process loading, the `Dataset` class is strictly focused on data representation and providing access to single data samples. 

From an object-oriented design perspective, the `Dataset` implements several key principles:
*   **Abstraction:** It abstracts away the specific details of the underlying data source. Whether the data is loaded from a CSV file, a JSON file, or generated dynamically from a NumPy array, the rest of the framework interacts with it uniformly.
*   **Single Responsibility Principle (SRP):** It is solely responsible for storing the data state and retrieving exactly one item at a time (along with any necessary single-item preprocessing), completely separating data access from data grouping.

A custom dataset class is created by subclassing this base `Dataset` and is strictly required to implement three main components: the `__init__` constructor, the `__len__` method, and the `__getitem__` method.

## \_\_init__()

The constructor initializes the dataset object by setting up the necessary attributes, file paths, or data structures that will be accessed later during data retrieval.

### Method Signature
```python
def __init__(self, features, labels, *args, **kwargs):
```

### Specification
*   **Inputs:** The exact inputs are highly flexible depending on the specific dataset being implemented. They commonly include:
    *   **`features`** / **`labels`**: Raw data arrays or lists (e.g., `X_train` and `y_train`).
    *   **File Paths**: Strings representing locations of data files (like `.csv` or `.txt` files) to be loaded into memory.
    *   **Configuration Objects**: Objects like tokenizers or max-length parameters for preprocessing text data.
*   **State:** The method must store these inputs as instance variables (e.g., `self.features`, `self.labels`) so they are preserved and accessible by the `__len__` and `__getitem__` methods. 

### Example Implementation (for a simple in-memory array dataset)
```python
def __init__(self, X, y):
    self.features = X
    self.labels = y
```

---

## \_\_len__()

This method allows the dataset object to respond to Python's built-in `len()` function.

### Method Signature
```python
def __len__(self):
```

### Specification
*   **Purpose:** Returns the total number of individual data samples contained within the dataset. 
*   **Importance:** The `DataLoader` relies on this method to determine the boundaries of the dataset, which is necessary for generating valid random indices during shuffling and for determining when an epoch is complete.
*   **Computation:** Typically returns the length of the primary data structure holding the features or labels (e.g., the number of rows in a feature array).

### Example Implementation
```python
def __len__(self):
    return self.labels.shape
```

---

## \_\_getitem__()

This method is the core engine of the `Dataset` class. It allows the dataset object to be indexed like a list or array (e.g., `dataset`), returning a single training example.

### Method Signature
```python
def __getitem__(self, index):
```

### Specification
*   **Inputs:** 
    *   **`index`**: An integer ranging from `0` to `len(self) - 1`. The `DataLoader` provides this index when assembling batches.
*   **Execution Logic:**
    1. Uses the provided `index` to look up the corresponding feature and label from the instance variables stored during `__init__`.
    2. Performs any necessary per-sample preprocessing (e.g., tokenizing a string, padding a sequence, or normalizing values).
*   **Returns:** A tuple containing exactly one input feature array and its corresponding target label array: `(x, y)`. In the mini-torch framework, these must be returned as standard NumPy arrays.

### Example Implementation
```python
def __getitem__(self, index):
    # Retrieve exactly one data record and the corresponding label
    one_x = self.features[index]
    one_y = self.labels[index]
    
    return one_x, one_y
```
