# Example concrete class that loads the MNIST dataset


import numpy as np
import csv
from Dataset import Dataset

class MNISTDataset(Dataset):
    """
    A Dataset subclass designed to load the Kaggle MNIST CSV files.
    It automatically normalizes pixel values and one-hot encodes labels 
    to support training.
    """

    def __init__(self, filepath, is_train=True, num_classes=10):
        """
        Initializes the dataset by reading the Kaggle CSV file into memory.

        Args:
            filepath (str): The path to the train.csv or test.csv file.
            is_train (bool): True if loading the training set (contains labels), 
                             False if loading the test set (no labels).
            num_classes (int): The number of categories (10 for MNIST).
        """
        raw_features = []
        raw_labels = []
        
        # Use standard python CSV reader to avoid bringing in Pandas as a dependency
        with open(filepath, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header row
            
            for row in reader:
                if is_train:
                    # train.csv format: Id (col 0), label (col 1), pixels (cols 2:786)
                    raw_labels.append(int(row[1]))
                    # Normalize pixel values to to prevent gradient saturation
                    raw_features.append([float(p) / 255.0 for p in row[2:]])
                else:
                    # test.csv format: Id (col 0), pixels (cols 1:785)
                    raw_labels.append(-1)  # Dummy label for the test set
                    raw_features.append([float(p) / 255.0 for p in row[1:]])
                # end if
            # end for
        # end with
        
        # Convert lists to framework-compatible NumPy arrays
        features_array = np.array(raw_features, dtype=np.float32)
        labels_array = np.array(raw_labels, dtype=int)
        
        # Initialize the abstract base class state
        super().__init__(features=features_array, labels=labels_array)
        
        self.is_train = is_train
        self.num_classes = num_classes
    # end method

    def __len__(self):
        """
        Returns the total number of individual image samples in the dataset.
        """
        return len(self.features)
    # end method

    def __getitem__(self, index):
        """
        Retrieves a single (feature, label) sample at the specified index.
        
        For Mean Squared Error (MSE) loss to work effectively on a classification 
        task, the target label must be returned as a one-hot encoded vector.

        Args:
            index (int): The index of the data sample to retrieve.

        Returns:
            tuple: A tuple (x, y) where x is the flattened pixel array (shape 784,)
                   and y is the one-hot encoded target array (shape 10,).
        """
        x = self.features[index]
        
        if self.is_train:
            # One-hot encode the target label for MSE loss
            label_idx = self.labels[index]
            y = np.zeros(self.num_classes, dtype=np.float32)
            y[label_idx] = 1.0
        else:
            # Return a zero vector as a dummy target for the test set
            y = np.zeros(self.num_classes, dtype=np.float32)
        # end if
            
        return x, y
    # end method

# end class
