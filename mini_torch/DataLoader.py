
import numpy as np

class DataLoader:
    """
    DataLoader class for the Mini-Torch framework.
    
    This class wraps a Dataset object to implement the Iterator Pattern, 
    handling the logic for grouping individual samples into minibatches 
    and optionally shuffling the data at the start of each epoch.
    """

    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False):
        """
        Initializes the DataLoader.

        Args:
            dataset (Dataset): An instance of a Dataset subclass.
            batch_size (int): The number of samples to group into a single batch.
            shuffle (bool): Whether to randomly shuffle the dataset indices per epoch.
            drop_last (bool): Whether to drop the final incomplete batch if the 
                              dataset size is not perfectly divisible by batch_size.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
    # end method

    def __iter__(self):
        """
        Transforms the DataLoader into a Python generator, yielding minibatches 
        of data one at a time for the training loop.
        
        Returns:
            tuple: A tuple containing (batch_x, batch_y) as batch-first NumPy arrays.
        """
        total_samples = len(self.dataset)
        indices = np.arange(total_samples)
        
        if self.shuffle:
            np.random.shuffle(indices)
        # end if
            
        for start_idx in range(0, total_samples, self.batch_size):
            batch_indices = indices[start_idx : start_idx + self.batch_size]
            
            # Handle the drop_last condition for incomplete batches
            if self.drop_last and len(batch_indices) < self.batch_size:
                break
            # end if
                
            batch_x = []
            batch_y = []
            
            for idx in batch_indices:
                x, y = self.dataset[idx]
                batch_x.append(x)
                batch_y.append(y)
            # end for
            
            # Stack the individual samples into unified row-vector NumPy arrays
            yield np.vstack(batch_x), np.vstack(batch_y)
        # end for
    # end method

    def __len__(self):
        """
        Returns the total number of batches that the data loader will yield per epoch.
        
        Returns:
            int: The total number of batches.
        """
        total_samples = len(self.dataset)
        
        if self.drop_last:
            return total_samples // self.batch_size
        else:
            return int(np.ceil(total_samples / self.batch_size))
        # end if
    # end method

# end class
