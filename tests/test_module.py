import pytest
import numpy as np
from mini_torch.Module import Module
from mini_torch.backend import xp, asnumpy

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
            self.dW = xp.array([1.5, -2.0, 3.1])
            
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
    np.testing.assert_array_equal(asnumpy(model.dW), np.array([0.0, 0.0, 0.0]))

def test_module_zero_grad_with_no_parameters_or_uninitialized_gradients():
    """Ensures zero_grad safely handles modules without parameters or with uninitialized gradients."""
    
    # 1. Test empty list case (e.g., Activation layers)
    class EmptyModule(Module):
        def forward(self, x): return x
        def backward(self, grad_output): return grad_output

    m_empty = EmptyModule()
    m_empty.zero_grad() # Should not raise an error

    # 2. Test uninitialized gradient case (e.g., Linear layer before backward pass)
    class UninitializedModule(Module):
        def __init__(self):
            super().__init__()
            self.dW = None
        def forward(self, x): return x
        def backward(self, grad_output): return grad_output
        def grads(self): return [self.dW]

    m_uninit = UninitializedModule()
    m_uninit.zero_grad() # Should not raise an AttributeError on None.fill()

def test_module_save_and_load_weights(tmp_path):
    """Tests the save_weights and load_weights functionality."""
    class DummyModule(Module):
        def __init__(self, w_val, b_val):
            super().__init__()
            self.W = xp.array([w_val, w_val], dtype=np.float32)
            self.b = xp.array([b_val], dtype=np.float32)
            
        def forward(self, x): return x
        def backward(self, grad_output): return grad_output
        def parameters(self): return [self.W, self.b]

    # Create original model and save weights
    model_orig = DummyModule(1.5, -0.5)
    filepath = tmp_path / "weights.npz"
    model_orig.save_weights(str(filepath))
    
    # Create new model with differently initialized weights
    model_new = DummyModule(0.0, 0.0)
    model_new.load_weights(str(filepath))
    
    # Assert weights were updated correctly from the file
    np.testing.assert_array_equal(asnumpy(model_new.W), np.array([1.5, 1.5], dtype=np.float32))
    np.testing.assert_array_equal(asnumpy(model_new.b), np.array([-0.5], dtype=np.float32))

def test_module_load_weights_shape_mismatch(tmp_path):
    """Ensures load_weights raises an assertion error if parameter shapes do not match."""
    class DummyModule(Module):
        def __init__(self, shape):
            super().__init__()
            self.W = xp.zeros(shape, dtype=np.float32)
            
        def forward(self, x): return x
        def backward(self, grad_output): return grad_output
        def parameters(self): return [self.W]
        
    model1 = DummyModule((2, 2))
    filepath = tmp_path / "weights_mismatch.npz"
    model1.save_weights(str(filepath))
    
    model2 = DummyModule((3, 3))
    
    with pytest.raises(AssertionError, match="Shape mismatch"):
        model2.load_weights(str(filepath))