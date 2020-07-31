from gans.callbacks import callback


class GlobalStepIncrementer(callback.Callback):

    def on_training_step_end(self, trainer):
        trainer.global_step += 1
