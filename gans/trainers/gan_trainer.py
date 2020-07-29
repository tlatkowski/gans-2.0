import os
from abc import abstractmethod
from typing import List

import numpy as np
from tqdm import tqdm

from gans.callbacks import callback
from gans.callbacks import logger
from gans.datasets import abstract_dataset
from gans.models import model
from gans.trainers import gan_checkpoint_manager as ckpt_manager
from gans.trainers import optimizers
from gans.utils import constants
from gans.utils import logging
from gans.utils import visualization

SEED = 0

log = logging.get_logger(__name__)


class GANTrainer:

    def __init__(
            self,
            batch_size: int,
            generators: List[model.Model],
            discriminators: List[model.Model],
            training_name: str,
            generators_optimizers: List[optimizers.Optimizer],
            discriminators_optimizers: List[optimizers.Optimizer],
            continue_training: bool,
            save_images_every_n_steps: int,
            num_test_examples=None,
            visualization_type: str = 'fn',
            checkpoint_step=10,
            save_model_every_n_step=1000,
            callbacks: List[callback.Callback] = None,
    ):
        self.batch_size = batch_size
        self.generators = generators
        self.discriminators = discriminators
        self.checkpoint_step = checkpoint_step
        self.save_model_every_n_step = save_model_every_n_step
        self.training_name = training_name
        self.save_images_every_n_steps = save_images_every_n_steps
        self.num_test_examples = num_test_examples
        self.visualization_type = visualization_type
        self.continue_training = continue_training
        self.callbacks = callbacks or []

        self.generators_optimizers = generators_optimizers
        self.discriminators_optimizers = discriminators_optimizers

        self.root_checkpoint_path = os.path.join(
            constants.SAVE_IMAGE_DIR,
            training_name,
        )
        self.logger = logger.TensorboardLogger(
            root_checkpoint_path=self.root_checkpoint_path,
        )
        self.checkpoint_manager = ckpt_manager.GANCheckpointManager(
            components_to_save={
                **self.generators_optimizers,
                **self.discriminators_optimizers,
                **{k: v.model for k, v in self.generators.items()},
                **{k: v.model for k, v in self.discriminators.items()}
            },
            root_checkpoint_path=self.root_checkpoint_path,
            continue_training=continue_training,
        )

    @abstractmethod
    def train_step(self, batch):
        raise NotImplementedError

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self):
        pass

    def on_training_step_begin(self):
        pass

    def on_training_step_end(self):
        for c in self.callbacks:
            c.on_training_step_end()

    def train(
            self,
            dataset: abstract_dataset.Dataset,
            num_epochs: int,
            validation_dataset: None,
    ):
        train_step = 0

        latest_checkpoint_epoch = self.checkpoint_manager.regenerate_training()
        latest_epoch = latest_checkpoint_epoch * self.checkpoint_step
        num_epochs += latest_epoch
        for epoch in tqdm(range(latest_epoch, num_epochs), desc='Epochs'):
            self.on_epoch_begin()
            dataset_tqdm = tqdm(
                iterable=dataset,
                desc="Batches",
                leave=True
            )
            for batch in dataset_tqdm:
                self.on_training_step_begin()
                losses = self.train_step(batch)
                self.on_training_step_end()
                self.logger.log_scalars(name='Losses', scalars=losses, step=train_step)

                if train_step % self.save_images_every_n_steps == 0:
                    for name, generator in self.generators.items():
                        if self.num_test_examples is None:
                            if isinstance(validation_dataset, list):
                                self.num_test_examples = validation_dataset[0].shape[0]
                            else:
                                self.num_test_examples = validation_dataset.shape[0]
                        if self.visualization_type == 'fn':
                            img_to_plot = visualization.generate_and_save_images_for_model_fn_problems(
                                generator_model=generator,
                                epoch=train_step,
                                test_input=validation_dataset,
                                training_name=self.training_name,
                                num_examples_to_display=self.num_test_examples,
                            )
                        elif self.visualization_type == 'image':
                            img_to_plot = visualization.generate_and_save_images_for_image_problems(
                                generator_model=generator,
                                epoch=train_step,
                                test_input=validation_dataset,
                                save_path=os.path.join(self.root_checkpoint_path, 'images'),
                                num_examples_to_display=self.num_test_examples,
                            )
                        else:
                            raise NotImplementedError
                        self.logger.log_images(
                            name='test_outputs',
                            images=np.reshape(img_to_plot, newshape=(1, 480, 640, 4)),
                            step=train_step,
                        )
                steps_per_second = 1. / dataset_tqdm.avg_time if dataset_tqdm.avg_time else 0.
                self.logger.log_scalars(name='', scalars={'steps_per_second': steps_per_second}, step=train_step)

                if train_step % self.save_model_every_n_step == 0:
                    self.checkpoint_manager.save(checkpoint_number=epoch)
                    log.info(f'Saved model for {train_step} step and {epoch} epoch.')
                train_step += 1
            self.on_epoch_end()
