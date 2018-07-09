import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod
import gym

class Model(ABC):

    def __init__(self, batch_size, x, y):

        self.batch_size = batch_size
        self.loss = None
        self.optimizer = None
        self.input = None
        self.output = None
        self.label = None
        self.train_step = None

        super(Model, self).__init__()

        self.set_data(x, y)

    @abstractmethod
    def set_data(self, x, y):
        pass

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def train(self, n_epochs):
        pass

    @abstractmethod
    def predict(self, x):
        pass

    def evaluate(self, val_data):
        pass

    def save(self, filename):
        pass






