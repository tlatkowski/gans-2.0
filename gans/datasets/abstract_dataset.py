import abc
from abc import abstractmethod


class Dataset(abc.ABC):

    def __init__(
            self,
            input_params,
            with_labels=False,
    ):
        self.batch_size = input_params.batch_size
        self.buffer_size = input_params.buffer_size
        if with_labels:
            self.train_dataset = self.load_data_with_labels()
        else:
            self.train_dataset = self.load_data()

    @abstractmethod
    def load_data(self):
        raise NotImplementedError

    @abstractmethod
    def load_data_with_labels(self):
        raise NotImplementedError

    def __iter__(self):
        return iter(self.train_dataset)
