import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.basemodel import Basemodel


class NeuralNetwork(torch.nn.Module, Basemodel):
    """
    Neural network model
    """

    def __init__(self, input_size=2, hidden_size=16, activation_fun=torch.nn.ReLU()):
        """
        Initialize the model
        :param input_size: input size
        :param hidden_size: hidden size
        :param activation_fun: activation function
        """
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation_fun = activation_fun

        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)  # output size is 1

    def forward(self, x):
        """
        Forward pass
        :param x: input
        """
        x = self.activation_fun(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze()

    def fit(self, X, y):
        """
        Fit the model
        :param X: training data
        :param y: training targets
        :return: self
        """
        self.input_size = X.shape[1]
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        n_samples = X.shape[0]
        y = y.iloc[-n_samples:]
        self._train(X, y)
        return self

    def predict(self, X):
        """
        Predict
        :param X: input
        :return: predictions
        """
        self.eval()
        X = torch.tensor(X.to_numpy(), dtype=torch.float32)
        with torch.no_grad():
            return self.forward(X).numpy()

    def _train(self, X, y, max_epochs=10, lr=0.01):
        """
        trains the model
        :param X: training data
        :param y: training targets
        :param max_epochs: maximum number of epochs
        :param lr: learning rate
        """
        self.train()
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        X = torch.tensor(X.to_numpy(), dtype=torch.float32)
        y = torch.tensor(y.to_numpy(), dtype=torch.float32)

        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

        for epoch in range(max_epochs):
            losses = np.array([])
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                output = self.forward(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                losses = np.append(losses, loss.item())
            print(f"Epoch: {epoch}, loss: {losses[-1]}")


class LSTM(torch.nn.Module):
    pass
