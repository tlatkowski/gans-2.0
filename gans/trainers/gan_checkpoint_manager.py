import os

import tensorflow as tf

from gans.utils import constants
from gans.utils import logging

logger = logging.get_logger(__name__)


class GANCheckpointManager:

    def __init__(
            self,
            components_to_save,
            root_checkpoint_path,
            continue_training,
    ):
        self.root_checkpoint_path = root_checkpoint_path
        self.continue_training = continue_training
        self.training_checkpoint_path = os.path.join(
            self.root_checkpoint_path,
            constants.CHECKPOINT_DIR,
        )
        self.checkpoint = tf.train.Checkpoint(
            **components_to_save,
        )
        self.checkpoint_manager = tf.train.CheckpointManager(
            checkpoint=self.checkpoint,
            directory=self.training_checkpoint_path,
            max_to_keep=3,
        )

    def load_for_predict(self):
        pass

    def load_for_train(self):
        pass

    def regenerate_training(self):
        latest_checkpoint_epoch = 0
        if self.continue_training:
            latest_checkpoint = self.checkpoint_manager.latest_checkpoint
            if latest_checkpoint is not None:
                latest_checkpoint_epoch = int(latest_checkpoint[latest_checkpoint.index("-") + 1:])
                self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
                logger.info(f'Training regeneration from checkpoint: {self.root_checkpoint_path}.')
            else:
                logger.info('No checkpoints found. Starting training from scratch.')
        return latest_checkpoint_epoch

    def save(self, checkpoint_number):
        self.checkpoint_manager.save(checkpoint_number=checkpoint_number)
