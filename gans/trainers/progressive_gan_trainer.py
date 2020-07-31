from gans.trainers import gan_trainer


class ProgressiveGANTrainer(gan_trainer.GANTrainer):

    def __init__(
            self,
            batch_size,
            generators,
            discriminators,
            training_name,
            generators_optimizers,
            discriminators_optimizers,
            continue_training,
            save_images_every_n_steps,
            checkpoint_step=10,
            validation_dataset=None,
            callbacks=None,
    ):
        super().__init__(
            batch_size=batch_size,
            generators=generators,
            discriminators=discriminators,
            training_name=training_name,
            generators_optimizers=generators_optimizers,
            discriminators_optimizers=discriminators_optimizers,
            continue_training=continue_training,
            save_images_every_n_steps=save_images_every_n_steps,
            checkpoint_step=checkpoint_step,
            validation_dataset=validation_dataset,
            callbacks=callbacks,
        )

    def train_step(self, batch):
        pass

    def test_inputs(self, dataset):
        pass
