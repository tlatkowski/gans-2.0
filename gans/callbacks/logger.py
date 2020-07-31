import abc

import tensorflow as tf


class Logger(abc.ABC):

    @abc.abstractmethod
    def log_scalars(self, name: str, scalars, step):
        pass

    @abc.abstractmethod
    def log_images(self, name: str, images, step):
        pass


class TensorboardLogger(Logger):

    def __init__(
            self,
            root_checkpoint_path,
    ):
        self.root_checkpoint_path = root_checkpoint_path
        self.summary_writer = tf.summary.create_file_writer(self.root_checkpoint_path)

    def log_scalars(self, name: str, scalars, step):
        with self.summary_writer.as_default():
            [tf.summary.scalar(f'{name}/{scalar_name}', v, step=step) for scalar_name, v in scalars.items()]

    def log_images(self, name: str, images, step):
        with self.summary_writer.as_default():
            tf.summary.image(name=name, data=images, step=step)
