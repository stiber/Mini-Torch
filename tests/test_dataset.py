import pytest
import numpy as np
import csv
from mini_torch.Dataset import Dataset
from examples.MNISTDataset import MNISTDataset

def test_mnist_dataset(tmp_path):
    """
    Tests the MNISTDataset implementation using a mock CSV file, 
    verifying the dataset length, feature shapes, normalization, 
    and one-hot encoding logic.
    """
    # 1. Create a dummy CSV to simulate 'train.csv' with 2 samples
    csv_file = tmp_path / "train.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Id', 'Label'] + [f'Pixel{i}' for i in range(784)])
        # Sample 0: Label 2, all zero pixels
        writer.writerow([0, 2] + [0] * 784)
        # Sample 1: Label 7, all zero pixels
        writer.writerow([1, 7] + [0] * 784)

    # 2. Instantiate the dataset
    dataset = MNISTDataset(filepath=str(csv_file), is_train=True)
    
    # 3. Assert the length of the dataset
    assert len(dataset) == 2
    
    # 4. Assert information about the selected images and their targets
    
    # --- Sample 0 ---
    x0, y0 = dataset[0]
    assert x0.shape == (784,)
    np.testing.assert_array_equal(x0[:5], np.array([0., 0., 0., 0., 0.], dtype=np.float32))
    assert y0.shape == (10,)
    
    expected_y0 = np.zeros(10, dtype=np.float32)
    expected_y0[2] = 1.0
    np.testing.assert_array_equal(y0, expected_y0)
    assert np.argmax(y0) == 2

    # --- Sample 1 ---
    x1, y1 = dataset[1]
    assert x1.shape == (784,)
    np.testing.assert_array_equal(x1[:5], np.array([0., 0., 0., 0., 0.], dtype=np.float32))
    assert y1.shape == (10,)
    
    expected_y1 = np.zeros(10, dtype=np.float32)
    expected_y1[7] = 1.0
    np.testing.assert_array_equal(y1, expected_y1)
    assert np.argmax(y1) == 7
#end function

def test_mnist_dataset_test(tmp_path):
    """
    Tests the MNISTDataset implementation using a mock CSV file, 
    verifying the dataset length, feature shapes, normalization, 
    and dummy label generation for test sets.
    """
    # 1. Create a dummy CSV to simulate 'test.csv' with 2 samples
    csv_file = tmp_path / "test.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Id'] + [f'Pixel{i}' for i in range(784)])
        # Sample 0: all zero pixels
        writer.writerow([0] + [0] * 784)
        # Sample 1: all zero pixels
        writer.writerow([1] + [0] * 784)

    # 2. Instantiate the dataset
    dataset = MNISTDataset(filepath=str(csv_file), is_train=False)
    
    # 3. Assert the length of the dataset
    assert len(dataset) == 2
    
    # 4. Assert information about the selected images and their targets
    
    # --- Sample 0 ---
    x0, y0 = dataset[0]
    assert x0.shape == (784,)
    np.testing.assert_array_equal(x0[:5], np.array([0., 0., 0., 0., 0.], dtype=np.float32))
    assert y0.shape == (10,)
    
    expected_y0 = np.zeros(10, dtype=np.float32)
    np.testing.assert_array_equal(y0, expected_y0)

    # --- Sample 1 ---
    x1, y1 = dataset[1]
    assert x1.shape == (784,)
    np.testing.assert_array_equal(x1[:5], np.array([0., 0., 0., 0., 0.], dtype=np.float32))
    assert y1.shape == (10,)
    
    expected_y1 = np.zeros(10, dtype=np.float32)
    np.testing.assert_array_equal(y1, expected_y1)
#end function
