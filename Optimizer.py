
import abc

class Optimizer(abc.ABC):
    """
    Abstract base class for all optimization algorithms in the Mini-Torch framework.
    
    This class handles the mathematical optimization of model parameters. By separating 
    the update mechanism from the model state, it implements the Strategy Pattern, 
    allowing students to seamlessly swap different optimization algorithms.
    """

    def __init__(self, modules, lr=0.01):
        """
        Initializes the Optimizer.

        Args:
            modules (list): A list of Module instances that make up the neural network.
            lr (float): The learning rate, controlling the step size during weight updates.
        """
        self.modules = modules
        self.lr = lr
    # end method

    def zero_grad(self):
        """
        Clears the gradient caches from the previous training step.
        
        Because the framework relies on manual backpropagation, gradients typically 
        accumulate by default. This method iterates through all registered modules 
        and explicitly resets their gradients to prevent unintended accumulation 
        across multiple batches.
        """
        for module in self.modules:
            if hasattr(module, 'zero_grad'):
                module.zero_grad()
            # end if
        # end for
    # end method

    @abc.abstractmethod
    def step(self):
        """
        Performs a single optimization step to mathematically update the network's weights.
        
        This is an abstract method. Every subclass (e.g., SGD, AdamW) must provide 
        its own implementation that iterates through the parameters and gradients 
        of each module to apply the specific update formula.
        """
        pass
    # end method

# end class
