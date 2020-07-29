import abc

import tensorflow as tf


class TensorboardLoggable(abc.ABC):

    @abc.abstractmethod
    def log_to_tensorboard(self, *args):
        pass


class StepsPerSecond(TensorboardLoggable):

    def log_to_tensorboard(
            self,
            dataset_tqdm,
            train_step,
    ):
        steps_per_second = 1. / dataset_tqdm.avg_time if dataset_tqdm.avg_time else 0.
        with self.summary_writer.as_default():
            tf.summary.scalar('steps_per_second', steps_per_second, train_step)
