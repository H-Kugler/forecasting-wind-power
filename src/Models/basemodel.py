from abc import ABC, abstractmethod

def Basemodel(ABC):
    
    @abstractmethod
    def predict(self, X, time_steps_ahead):
        """
        Predicts the next Power output for the given data time_steps_ahead into the future.
        :param data: The data to predict on
        :param time_steps_ahead: The number of time steps into the future to predict
        :return: A list of predictions
        """
        pass
    