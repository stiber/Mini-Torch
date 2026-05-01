
import abc

class Dataset(abc.ABC):
    """
    Abstract base class for all datasets in the Mini-Torch framework.
    
    This class abstractions away the specific details of the underlying data source 
    (e.g., NumPy arrays, CSV files). It strictly adheres to the Single Responsibility 
    Principle by focusing solely on storing the data state and retrieving exactly one 
    item at a time, leaving batching and shuffling to the DataLoader.
    """

    def __init__(self, features, labels, *args, **kwargs):
        """
        Initializes the dataset object by storing the primary data structures.

        Args:
            features: The raw input data, array structures, or file paths.
            labels: The target data corresponding to the features.
            *args: Variable positional arguments for subclass flexibility.
            **kwargs: Variable keyword arguments for subclass flexibility (e.g., tokenizers).
        """
        self.features = features
        self.labels = labels
    # end method

    @abc.abstractmethod
    def __len__(self):
        """
        Returns the total number of individual data samples contained within the dataset.
        
        Subclasses must implement this method. The DataLoader relies on it to determine 
        the boundaries of the dataset for generating random indices during shuffling 
        and figuring out when a training epoch is complete.

        Returns:
            int: The total number of data records.
        """
        pass
    # end method

    @abc.abstractmethod
    def __getitem__(self, index):
        """
        Retrieves a single data sample and its corresponding label at the specified index.
        
        Subclasses must implement this method. It is responsible for looking up the 
        feature and label from the instance variables and performing any necessary 
        per-sample preprocessing (e.g., tokenization or padding) before returning them.

        Args:
            index (int): An integer ranging from 0 to len(self) - 1.

        Returns:
            tuple: A tuple (x, y) containing the input feature array and target label array.
        """
        pass
    # end method

# end class
