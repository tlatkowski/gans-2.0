import abc


class Callback(abc.ABC):

    @abc.abstractmethod
    def on_epoch_begin(self):
        pass

    @abc.abstractmethod
    def on_epoch_end(self):
        pass

    @abc.abstractmethod
    def on_training_step_begin(self):
        pass

    @abc.abstractmethod
    def on_training_step_end(self):
        pass
