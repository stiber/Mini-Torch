
import abc

# Assuming Module is already defined/imported from the framework
# class Activation(Module): 
class Activation(Module, abc.ABC):
    """
    Abstract base class for all activation functions in the Mini-Torch framework.
    
    This class specializes the Module interface for element-wise, non-linear 
    transformations. It enforces the rule that activation functions contain 
    no trainable parameters (no weights or biases) and establishes the caching 
    state needed for manual backpropagation.
    """

    def __init__(self):
        """
        Initializes the Activation module.
        
        Calls the superclass constructor and sets up a single placeholder 
        variable to cache the input data during the forward pass.
        """
        super().__init__()
        self.x = None
    # end method

    @abc.abstractmethod
    def forward(self, x):
        """
        Performs the forward pass computation.
        
        Subclasses must implement this method to apply their specific non-linear 
        equation element-wise across the input array. They must also cache 
        the input (e.g., self.x = x) before returning the transformed array.

        Args:
            x (numpy.ndarray): The input data batch.

        Returns:
            numpy.ndarray: The transformed output data.
        """
        pass
    # end method

    @abc.abstractmethod
    def backward(self, grad_output):
        """
        Performs the backward pass computation (manual backpropagation).
        
        Subclasses must implement this method to compute the local derivative 
        using the cached self.x, and then multiply it element-wise by the 
        incoming grad_output to apply the chain rule.

        Args:
            grad_output (numpy.ndarray): The gradient of the loss with respect 
                                         to this layer's output.

        Returns:
            numpy.ndarray: The gradient of the loss with respect to this layer's input.
        """
        pass
    # end method

    def parameters(self):
        """
        Retrieves the module's learnable parameters.
        
        Overrides the base method to explicitly return an empty list, ensuring 
        the Optimizer does not attempt to update non-existent weights.

        Returns:
            list: An empty list [].
        """
        return []
    # end method

    def grads(self):
        """
        Retrieves the module's cached gradients.
        
        Overrides the base method to explicitly return an empty list, ensuring 
        the Optimizer does not attempt to look for parameter gradients.

        Returns:
            list: An empty list [].
        """
        return []
    # end method

# end class
