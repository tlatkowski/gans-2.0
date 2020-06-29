from abc import ABC
from abc import abstractmethod

from easydict import EasyDict as edict
from tensorflow.python import keras


class Model(ABC):

    def __init__(
            self,
            model_parameters: edict = None,
    ):
        self._model_parameters = model_parameters
        self._model = self.define_model()

    def __call__(self, inputs, **kwargs):
        return self.model(inputs=inputs, **kwargs)

    @abstractmethod
    def define_model(self) -> keras.Model:
        raise NotImplementedError

    @property
    def trainable_variables(self):
        return self.model.trainable_variables

    @property
    def model(self):
        return self._model

    @property
    def model_parameters(self) -> edict:
        return self._model_parameters

    @property
    def num_channels(self) -> int:
        return self.model.output_shape[-1]

    @property
    def model_name(self) -> str:
        return self.__class__.__name__

    def __repr__(self):
        return self.model_name
