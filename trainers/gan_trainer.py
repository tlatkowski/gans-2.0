import os
from abc import abstractmethod

import tensorflow as tf

from datasets import abstract_dataset
from utils import constants

SEED = 0


class GANTrainer:
    
    def __init__(
            self,
            batch_size,
            generator,
            discriminator,
            dataset_type,
            lr_generator,
            lr_discriminator,
            continue_training,
            save_images_every_n_steps,
            checkpoint_step=10,
    ):
        self.batch_size = batch_size
        self.generator = generator
        self.discriminator = discriminator
        self.checkpoint_step = checkpoint_step
        self.dataset_type = dataset_type
        self.lr_generator = lr_generator
        self.lr_discriminator = lr_discriminator
        self.save_images_every_n_steps = save_images_every_n_steps
        self.continue_training = continue_training
        
        self.generator_optimizer = tf.keras.optimizers.Adam(self.lr_generator, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(self.lr_discriminator, beta_1=0.5)
        
        self.checkpoint_path = os.path.join(
            constants.SAVE_IMAGE_DIR,
            dataset_type,
            constants.CHECKPOINT_DIR,
        )
        
        self.checkpoint_prefix = os.path.join(self.checkpoint_path, "ckpt")
        self.discriminator = self.discriminator
        self.generator = self.generator
        self.checkpoint = tf.train.Checkpoint(
            generator_optimizer_f=self.generator_optimizer,
            discriminator_optimizer_x=self.discriminator_optimizer,
            generator=self.generator.model,
            discriminator=self.discriminator.model,
        )
        self.summary_writer = tf.summary.create_file_writer(self.checkpoint_path)
    
    @abstractmethod
    def train(
            self,
            dataset: abstract_dataset.Dataset,
            num_epochs: int,
    ):
        raise NotImplementedError
