This software is free for non-commercial use under a CC BY-NC 4.0 license. For commercial licensing inquiries, please contact [Michael Stiber](mailto:stiber@uw.edu).

# Mini-Torch
A Python framework designed for computer science students building neural networks from scratch.

This repository contains and documents the architecture for a "Mini-Torch" framework, designed for computer science students building neural networks from scratch. It mirrors the PyTorch API using only `numpy`, `matplotlib`, select elements of `scipy`, and standard Python, emphasizing manual gradient calculations and batch-first row-vector notation.

## Package Usage

After installing the package (for example with `pip install -e .`), you can import core components using the `mini_torch` package namespace:

```python
from mini_torch import Module, Dataset
from mini_torch import DataLoader, Loss, Optimizer, Activation

# or import a submodule directly:
import mini_torch.Module
```


To improve encapsulation and modularity, this framework incorporates the following major architectural elements: an `Optimizer` base class (parameter updating), a `Loss` base class (error and initial gradient calculation), a `Dataset`/`DataLoader` pipeline (base classes to structure and iterate through data), a `Module` base class (layers, forward and backward passes), and a `Sequential` container (`Module` subclass that manages chaining of multiple layers).

---

*This work has been partially generated with use of a language model (AI), and the author has read through and tested the resulting content to ensure it accurately reflects the original intent. All errors are the responsibility of the author.*
