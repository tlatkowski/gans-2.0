import os

import numpy as np

from gans.callbacks import callback
from gans.utils import visualization


class ImageProblemSaver(callback.Callback):

    def __init__(
            self,
            save_images_every_n_steps: int,
            num_test_examples: int = None,
    ):
        self.save_images_every_n_steps = save_images_every_n_steps
        self.num_test_examples = num_test_examples

    def on_training_step_end(self, trainer):
        if trainer.global_step % self.save_images_every_n_steps == 0:
            for name, generator in trainer.generators.items():
                if self.num_test_examples is None:
                    if isinstance(trainer.validation_dataset, list):
                        self.num_test_examples = trainer.validation_dataset[0].shape[0]
                    else:
                        self.num_test_examples = trainer.validation_dataset.shape[0]

                img_to_plot = visualization.generate_and_save_images_for_image_problems(
                    generator_model=generator,
                    epoch=trainer.global_step,
                    test_input=trainer.validation_dataset,
                    save_path=os.path.join(trainer.root_checkpoint_path, 'images'),
                    num_examples_to_display=self.num_test_examples,
                )
                trainer.logger.log_images(
                    name='test_outputs',
                    images=np.reshape(img_to_plot, newshape=(1, 480, 640, 4)),
                    step=trainer.global_step,
                )


class FunctionProblemSaver(callback.Callback):

    def __init__(
            self,
            save_images_every_n_steps: int,
            num_test_examples: int = None,
    ):
        self.save_images_every_n_steps = save_images_every_n_steps
        self.num_test_examples = num_test_examples

    def on_training_step_end(self, trainer):
        if trainer.global_step % self.save_images_every_n_steps == 0:
            for name, generator in trainer.generators.items():
                if self.num_test_examples is None:
                    if isinstance(trainer.validation_dataset, list):
                        self.num_test_examples = trainer.validation_dataset[0].shape[0]
                    else:
                        self.num_test_examples = trainer.validation_dataset.shape[0]
                img_to_plot = visualization.generate_and_save_images_for_model_fn_problems(
                    generator_model=generator,
                    epoch=trainer.global_step,
                    test_input=trainer.validation_dataset,
                    training_name=trainer.training_name,
                    num_examples_to_display=self.num_test_examples,
                )

                trainer.logger.log_images(
                    name='test_outputs',
                    images=np.reshape(img_to_plot, newshape=(1, 480, 640, 4)),
                    step=trainer.global_step,
                )
