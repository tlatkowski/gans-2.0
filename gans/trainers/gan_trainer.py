import os
from abc import abstractmethod

import numpy as np
import tensorflow as tf

from gans.datasets import abstract_dataset
from gans.utils import constants
from gans.utils import logging
from gans.utils import visualization

SEED = 0

logger = logging.get_logger(__name__)


class GANTrainer:

    def __init__(
            self,
            batch_size,
            generators,
            discriminators,
            dataset_type,
            generators_optimizers,
            discriminators_optimizers,
            continue_training,
            save_images_every_n_steps,
            num_test_examples=None,
            visualization_type='fn',
            checkpoint_step=10,
    ):
        self.batch_size = batch_size
        self.generators = generators
        self.discriminators = discriminators
        self.checkpoint_step = checkpoint_step
        self.dataset_type = dataset_type
        self.save_images_every_n_steps = save_images_every_n_steps
        self.num_test_examples = num_test_examples
        self.visualization_type = visualization_type
        self.continue_training = continue_training

        self.generators_optimizers = generators_optimizers
        self.discriminators_optimizers = discriminators_optimizers

        self.checkpoint_path = os.path.join(
            constants.SAVE_IMAGE_DIR,
            dataset_type,
            constants.CHECKPOINT_DIR,
        )

        self.checkpoint_prefix = os.path.join(self.checkpoint_path, "ckpt")
        self.checkpoint = tf.train.Checkpoint(
            **{k: v for k, v in self.generators_optimizers.items()},
            **{k: v for k, v in self.discriminators_optimizers.items()},
            **{k: v.model for k, v in self.generators.items()},
            **{k: v.model for k, v in self.discriminators.items()}
        )
        self.summary_writer = tf.summary.create_file_writer(self.checkpoint_path)

    @abstractmethod
    def train_step(self, batch):
        raise NotImplementedError

    @abstractmethod
    def test_inputs(self, dataset):
        raise NotImplementedError

    def train(
            self,
            dataset: abstract_dataset.Dataset,
            num_epochs: int,
    ):
        train_step = 0
        test_samples = self.test_inputs(dataset)

        latest_checkpoint_epoch = self.regenerate_training()
        latest_epoch = latest_checkpoint_epoch * self.checkpoint_step
        num_epochs += latest_epoch
        for epoch in range(latest_epoch, num_epochs):
            for batch in dataset:
                losses = self.train_step(batch)
                with self.summary_writer.as_default():
                    [tf.summary.scalar(f'Losses/{loss_name}', v, step=train_step) for loss_name, v in losses.items()]

                if train_step % self.save_images_every_n_steps == 0:
                    for name, generator in self.generators.items():
                        if self.num_test_examples is None:
                            if isinstance(test_samples, list):
                                self.num_test_examples = test_samples[0].shape[0]
                            else:
                                self.num_test_examples = test_samples.shape[0]
                        if self.visualization_type == 'fn':
                            img_to_plot = visualization.generate_and_save_images_for_model_fn_problems(
                                generator_model=generator,
                                epoch=train_step,
                                test_input=test_samples,
                                dataset_name=self.dataset_type,
                                num_examples_to_display=self.num_test_examples,
                            )
                        elif self.visualization_type == 'image':
                            img_to_plot = visualization.generate_and_save_images_for_image_problems(
                                generator_model=generator,
                                epoch=train_step,
                                test_input=test_samples,
                                dataset_name=self.dataset_type,
                                num_examples_to_display=self.num_test_examples,
                            )
                        else:
                            raise NotImplementedError
                        with self.summary_writer.as_default():
                            tf.summary.image(
                                name='test_outputs',
                                data=np.reshape(img_to_plot, newshape=(1, 480, 640, 4)),
                                step=train_step,
                            )

                train_step += 1

    def regenerate_training(self):
        latest_checkpoint_epoch = 0
        if self.continue_training:
            latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_path)
            if latest_checkpoint is not None:
                latest_checkpoint_epoch = int(latest_checkpoint[latest_checkpoint.index("-") + 1:])
                self.checkpoint.restore(latest_checkpoint)
                logger.info(f'Training regeneration from checkpoint: {self.checkpoint_path}.')
            else:
                logger.info('No checkpoints found. Starting training from scratch.')
        return latest_checkpoint_epoch
