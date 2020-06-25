from abc import ABC
from abc import abstractmethod

from easydict import EasyDict as edict


class Generator(ABC):

    def __init__(
            self,
            model_parameters: edict,
    ):
        self._model_parameters = model_parameters
        self._model = self.define_model()

    def __call__(self, inputs, **kwargs):
        return self._model(inputs=inputs, **kwargs)

    @abstractmethod
    def define_model(self):
        raise NotImplementedError

    @property
    def trainable_variables(self):
        return self.model.trainable_variables

    @property
    def model(self):
        return self._model

    @property
    def num_channels(self):
        return self.model.output_shape[-1]

    def __repr__(self):
        return self.__name__
