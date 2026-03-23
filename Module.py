
import abc

class Module(abc.ABC):
    """
    Abstract base class for all neural network layers, activation functions, 
    and containers in the Mini-Torch framework.
    
    This class defines the standard interface that all network components 
    must share, ensuring uniformity and interchangeability (the Composite Pattern) 
    throughout the framework.
    """

    def __init__(self):
        """
        Initializes the Module. 
        
        Subclasses should call super().__init__() and define their specific 
        trainable parameters (e.g., self.W, self.b) and gradient caches 
        (e.g., self.x, self.dW, self.db) within their own constructors.
        """
        pass
    # end method

    @abc.abstractmethod
    def forward(self, x):
        """
        Performs the forward pass computation.
        
        Subclasses must implement this method to apply their specific mathematical 
        operations to the input. They must also cache the input data here if it 
        is required to compute local derivatives during the backward pass.

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
        
        Subclasses must implement this method to calculate how the loss changes 
        with respect to their parameters and inputs. They should update their 
        internal gradient caches (e.g., self.dW, self.db) and return the gradient 
        to be passed to the preceding layer.

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
        
        By default, this returns an empty list, which is the correct behavior 
        for parameter-free modules like Activation layers. Layers with weights 
        (like Linear) or containers (like Sequential) must override this method.

        Returns:
            list: A list of NumPy arrays representing the trainable parameters.
        """
        return []
    # end method

    def grads(self):
        """
        Retrieves the module's cached gradients.
        
        By default, this returns an empty list. Subclasses with trainable 
        parameters must override this to return a list of gradients that perfectly 
        aligns 1-to-1 with the list returned by parameters().

        Returns:
            list: A list of NumPy arrays representing the parameter gradients.
        """
        return []
    # end method

# end class
