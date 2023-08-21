import numpy as np
import pandas as pd


def MovingAverage(basemodel):

    def __init__(self, window_size):
        self.name = "Moving Average"
        self.window_size = window_size

    def predict(self, data, time_steps_ahead):
        pass