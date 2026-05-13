import pytest
from mini_torch.Optimizer import Optimizer

def test_optimizer_cannot_be_instantiated_directly():
    """Ensures that Python prevents direct instantiation of the ABC."""
    with pytest.raises(TypeError):
        opt = Optimizer(modules=[])

def test_optimizer_concrete_methods():
    """
    Creates a DummyOptimizer to test the inherited concrete 
    method: zero_grad().
    """
    class MockModule:
        def __init__(self):
            self.zero_grad_called = False
            
        def zero_grad(self):
            self.zero_grad_called = True

    class DummyOptimizer(Optimizer):
        def step(self):
            pass

    mod1 = MockModule()
    mod2 = MockModule()
    
    # Module without zero_grad to ensure the hasattr check prevents an AttributeError
    class NoZeroGradModule:
        pass
    mod3 = NoZeroGradModule()

    # Initialize dummy optimizer with the mock modules
    opt = DummyOptimizer(modules=[mod1, mod2, mod3], lr=0.05)
    
    # Test state initialization
    assert opt.lr == 0.05
    assert opt.modules == [mod1, mod2, mod3]
    
    # Test zero_grad delegation
    opt.zero_grad()
    
    # Assert the method delegated the zero_grad call down to the children
    assert mod1.zero_grad_called is True
    assert mod2.zero_grad_called is True