import abc


class Callback(abc.ABC):

    def on_epoch_begin(self, trainer):
        pass

    def on_epoch_end(self, trainer):
        pass

    def on_training_step_begin(self, trainer):
        pass

    def on_training_step_end(self, trainer):
        pass
