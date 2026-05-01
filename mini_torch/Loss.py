
import abc

class Loss(abc.ABC):
    """
    Abstract base class for all loss functions in the Mini-Torch framework.
    
    This class implements the Strategy Pattern for error calculation. It requires 
    subclasses to compute a scalar performance metric and the initial gradient 
    to kick off the backpropagation process. It also manages the caching of 
    predictions and targets required for manual gradient computation.
    """

    def __init__(self):
        """
        Initializes the Loss module.
        
        Sets up placeholder variables to cache the network's predictions and 
        the true target values during the forward pass, which are required 
        subsequently during the backward pass.
        """
        self.predictions = None
        self.targets = None
    # end method

    @abc.abstractmethod
    def forward(self, predictions, targets):
        """
        Computes the scalar loss metric.
        
        Subclasses must implement this method to calculate the specific error 
        (e.g., Mean Squared Error, Cross-Entropy). They must also cache the 
        passed predictions and targets in self.predictions and self.targets 
        before returning the calculated scalar value.

        Args:
            predictions (numpy.ndarray): The output from the neural network.
            targets (numpy.ndarray): The ground truth labels or values.

        Returns:
            float: The calculated scalar loss.
        """
        pass
    # end method

    @abc.abstractmethod
    def backward(self):
        """
        Computes the gradient of the loss with respect to the predictions.
        
        Subclasses must implement this method using the cached self.predictions 
        and self.targets to compute the initial grad_output array. This array 
        will then be passed backward into the neural network's final layer.

        Returns:
            numpy.ndarray: The gradient of the loss with respect to predictions.
        """
        pass
    # end method

# end class
