import pytest
import numpy as np

@pytest.fixture
def sample_batch():
    """Provides a standard 2D numpy array for forward/backward pass testing."""
    return np.array([[1.0, -2.0, 3.0], [-1.0, 5.0, -0.5]], dtype=np.float32)