import pytest
from mini_torch.Activation import Activation

def test_activation_cannot_be_instantiated_directly():
    """Ensures that Activation strictly requires forward/backward overrides."""
    with pytest.raises(TypeError):
        act = Activation()

def test_activation_missing_abstract_methods():
    """Ensures that a subclass failing to implement all abstract methods cannot be instantiated."""
    class IncompleteActivation(Activation):
        def forward(self, x):
            pass
        # Missing backward()
            
    with pytest.raises(TypeError):
        act = IncompleteActivation()

def test_activation_concrete_methods():
    """
    Creates a dummy activation to test that the specialized parameters() 
    and grads() overrides correctly return empty lists, and that state is initialized.
    """
    class DummyActivation(Activation):
        def forward(self, x):
            self.x = x
            return x
            
        def backward(self, grad_output):
            return grad_output

    act = DummyActivation()
    
    # Ensure the caching property was initialized by super().__init__()
    assert act.x is None
    
    assert act.parameters() == []
    assert act.grads() == []