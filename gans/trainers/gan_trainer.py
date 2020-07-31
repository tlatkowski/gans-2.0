import os
from abc import abstractmethod
from typing import List

from tqdm import tqdm

from gans.callbacks import basic_callbacks
from gans.callbacks import callback
from gans.callbacks import logger
from gans.callbacks import saver
from gans.datasets import abstract_dataset
from gans.models import model
from gans.trainers import gan_checkpoint_manager as ckpt_manager
from gans.trainers import optimizers
from gans.utils import constants
from gans.utils import logging

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
            save_model_every_n_step=100,
            callbacks: List[callback.Callback] = None,
            validation_dataset=None,
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
        self.validation_dataset = validation_dataset

        self.global_step = 0
        self.epoch = 0

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
        self.saver = saver.Saver(
            save_images_every_n_steps=save_images_every_n_steps,
            num_test_examples=num_test_examples,
        )
        self.callbacks = callbacks or [
            self.checkpoint_manager,
            self.saver,
            basic_callbacks.GlobalStepIncrementer(),
        ]

    @abstractmethod
    def train_step(self, batch):
        raise NotImplementedError

    def train(
            self,
            dataset: abstract_dataset.Dataset,
            num_epochs: int,
    ):
        global_step = 0
        dataset_tqdm = tqdm(
            iterable=dataset,
            desc="Batches",
            leave=True
        )

        latest_checkpoint_epoch = self.checkpoint_manager.regenerate_training()
        latest_epoch = latest_checkpoint_epoch * self.checkpoint_step
        num_epochs += latest_epoch
        for self.epoch in tqdm(range(latest_epoch, num_epochs), desc='Epochs'):
            self.on_epoch_begin()
            for batch in dataset_tqdm:
                self.on_training_step_begin()
                losses = self.train_step(batch)
                self.on_training_step_end()
                self.logger.log_scalars(name='Losses', scalars=losses, step=global_step)
                steps_per_second = 1. / dataset_tqdm.avg_time if dataset_tqdm.avg_time else 0.
                self.logger.log_scalars(name='', scalars={'steps_per_second': steps_per_second}, step=self.global_step)
            self.on_epoch_end()

    def on_epoch_begin(self):
        for c in self.callbacks:
            c.on_epoch_begin(self)

    def on_epoch_end(self):
        for c in self.callbacks:
            c.on_epoch_end(self)

    def on_training_step_begin(self):
        for c in self.callbacks:
            c.on_training_step_begin(self)

    def on_training_step_end(self):
        for c in self.callbacks:
            c.on_training_step_end(self)
