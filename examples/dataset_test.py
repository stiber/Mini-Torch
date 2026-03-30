
import numpy as np
import sys
from pathlib import Path

# Add the parent directory (src) to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Now you can import from the src folder
from Dataset import Dataset
from MNISTDataset import MNISTDataset

# Assuming the MNISTDataset class (and the Dataset base class) 
# have been defined or imported here.

def test_mnist_dataset():
    # 1. Instantiate the dataset
    # We assume 'train.csv' is located in the same directory as the script.
    print("Loading dataset from 'train.csv'...")
    dataset = MNISTDataset(filepath='MNIST/train.csv', is_train=True)
    
    # 2. Output the length of the dataset
    print(f"\nTotal samples in the dataset: {len(dataset)}")
    
    # 3. Output information about a couple of selected images and their targets
    # We will iterate through the first two samples
    for i in range(2):
        x, y = dataset[i]
        
        print(f"\n--- Sample {i} ---")
        
        # Displaying feature (image) information
        print(f"Feature array shape: {x.shape}")
        # Show a small slice of the pixel array to verify they are normalized between 0.0 and 1.0
        print(f"First 5 normalized pixel values: {x[:5]}") 
        
        # Displaying target (label) information
        print(f"Target array shape: {y.shape}")
        print(f"Target vector (one-hot encoded): {y}")
        
        # Use np.argmax to convert the one-hot encoded vector back to the original integer class
        original_class = np.argmax(y)
        print(f"Associated target class (integer 0-9): {original_class}")
    #end for
#end method

if __name__ == "__main__":
    test_mnist_dataset()

### Expected Output
# When you run this script with the Kaggle MNIST `train.csv` file in your directory, it will parse the file and output something similar to this:

# ```text
# Loading dataset from 'train.csv'...

# Total samples in the dataset: 60000

# --- Sample 0 ---
# Feature array shape: (784,)
# First 5 normalized pixel values: [0. 0. 0. 0. 0.]
# Target array shape: (10,)
# Target vector (one-hot encoded): [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
# Associated target class (integer 0-9): 2

# --- Sample 1 ---
# Feature array shape: (784,)
# First 5 normalized pixel values: [0. 0. 0. 0. 0.]
# Target array shape: (10,)
# Target vector (one-hot encoded): [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
# Associated target class (integer 0-9): 7
