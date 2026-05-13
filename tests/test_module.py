import pytest
import numpy as np
from mini_torch.Module import Module

def test_module_cannot_be_instantiated_directly():
    """Ensures that Python prevents direct instantiation of the ABC."""
    with pytest.raises(TypeError):
        m = Module()

def test_module_missing_abstract_methods():
    """Ensures that a subclass failing to implement all abstract methods cannot be instantiated."""
    class IncompleteModule(Module):
        def forward(self, x):
            pass
        # Missing backward()
            
    with pytest.raises(TypeError):
        m = IncompleteModule()

def test_module_default_and_concrete_methods():
    """
    Creates a valid DummyModule to test the inherited concrete 
    methods: parameters(), grads(), and zero_grad().
    """
    class DummyModule(Module):
        def __init__(self):
            super().__init__()
            # Simulate a gradient array that needs to be zeroed
            self.dW = np.array([1.5, -2.0, 3.1])
            
        def forward(self, x):
            return x
            
        def backward(self, grad_output):
            return grad_output
            
        def grads(self):
            return [self.dW]
            
    # Instantiate the dummy implementation
    model = DummyModule()
    
    # Test the default base implementation of parameters()
    assert model.parameters() == []
    
    # Test the concrete implementation of zero_grad()
    model.zero_grad()
    np.testing.assert_array_equal(model.dW, np.array([0.0, 0.0, 0.0]))