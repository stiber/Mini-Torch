import pytest
import numpy as np
from mini_torch.DataLoader import DataLoader
from mini_torch.Dataset import Dataset

# Create a simple, isolated mock dataset for testing
class DummyDataset(Dataset):
    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

@pytest.fixture
def dummy_dataset():
    """Provides a basic dataset with 10 sequential samples."""
    features = np.arange(10, dtype=np.float32).reshape(10, 1)
    labels = np.arange(10, dtype=np.float32).reshape(10, 1)
    return DummyDataset(features, labels)

def test_dataloader_basic_batching(dummy_dataset):
    """Tests that the dataloader correctly chunks data into batches."""
    dataloader = DataLoader(dummy_dataset, batch_size=3, shuffle=False, drop_last=False)
    
    # 10 samples / 3 batch size = 4 batches (sizes: 3, 3, 3, 1)
    assert len(dataloader) == 4
    
    batches = list(dataloader)
    assert len(batches) == 4
    
    # Check shapes
    assert batches[0][0].shape == (3, 1) # First batch x
    assert batches[3][0].shape == (1, 1) # Last batch x
    
    # Verify values are correctly vertically stacked and in order
    np.testing.assert_array_equal(batches[0][0], np.array([[0.], [1.], [2.]]))
    np.testing.assert_array_equal(batches[0][1], np.array([[0.], [1.], [2.]]))

def test_dataloader_drop_last(dummy_dataset):
    """Tests that the drop_last parameter successfully discards incomplete batches."""
    dataloader = DataLoader(dummy_dataset, batch_size=3, shuffle=False, drop_last=True)
    
    # 10 samples / 3 batch size = 3 full batches, 1 dropped
    assert len(dataloader) == 3
    
    batches = list(dataloader)
    assert len(batches) == 3
    
    # The last batch of size 1 should no longer exist
    assert batches[-1][0].shape == (3, 1)

def test_dataloader_shuffle(dummy_dataset):
    """Tests that shuffling correctly randomizes the sequence of the dataset."""
    dataloader = DataLoader(dummy_dataset, batch_size=10, shuffle=True, drop_last=False)
    
    batches = list(dataloader)
    batch_x, batch_y = batches[0]
    
    # Check that data wasn't lost or duplicated
    assert np.sum(batch_x) == np.sum(dummy_dataset.features)
    
    # Ensure the arrays are no longer strictly in their original sequential order
    assert not np.array_equal(batch_x, dummy_dataset.features)