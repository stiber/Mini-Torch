import pytest
import numpy as np
import csv
from examples.MNISTDataset import MNISTDataset

def test_mnist_dataset_train(tmp_path):
    """Test loading a mock training CSV file."""
    # Setup dummy training data
    csv_file = tmp_path / "mock_train.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Id', 'Label'] + [f'Pixel{i}' for i in range(784)])
        # Create a dummy row for training: Label 4, and some pixel values
        row1 = [100, 4] + [0] * 784
        row1[2] = 255 # Make one pixel white (index 0 of pixel data)
        row1[3] = 127 # Make another gray (index 1 of pixel data)
        writer.writerow(row1)
        
    dataset = MNISTDataset(filepath=str(csv_file), is_train=True, num_classes=10)
    
    # Verify dataset boundaries
    assert len(dataset) == 1
    x, y = dataset[0]
    
    assert x.shape == (784,)
    assert y.shape == (10,)
    
    # Check one-hot encoding for label 4
    expected_y = np.zeros(10, dtype=np.float32)
    expected_y[4] = 1.0
    np.testing.assert_array_equal(y, expected_y)
    
    # Check normalization (255/255 = 1.0, 127/255 = ~0.498)
    assert x[0] == 1.0
    assert np.isclose(x[1], 127 / 255.0)

def test_mnist_dataset_test(tmp_path):
    """Test loading a mock testing CSV file."""
    # Setup dummy test data
    csv_file = tmp_path / "mock_test.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Id'] + [f'Pixel{i}' for i in range(784)])
        # Create a dummy row for testing: no label column
        row1 = [101] + [0] * 784
        row1[1] = 255 # index 0 of pixel data
        writer.writerow(row1)
        
    dataset = MNISTDataset(filepath=str(csv_file), is_train=False, num_classes=10)
    
    # Verify dataset boundaries
    assert len(dataset) == 1
    x, y = dataset[0]
    
    assert x.shape == (784,)
    assert y.shape == (10,)
    
    # Dummy target for test sets is expected to be all zeros
    np.testing.assert_array_equal(y, np.zeros(10, dtype=np.float32))
    
    # Check normalization
    assert x[0] == 1.0