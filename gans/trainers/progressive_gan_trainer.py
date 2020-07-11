from gans.trainers import gan_trainer


class ProgressiveGANTrainer(gan_trainer.GANTrainer):

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
            visualization_type: str,
            checkpoint_step=10,
    ):
        super().__init__(
            batch_size=batch_size,
            generators=generators,
            discriminators=discriminators,
            dataset_type=dataset_type,
            generators_optimizers=generators_optimizers,
            discriminators_optimizers=discriminators_optimizers,
            continue_training=continue_training,
            save_images_every_n_steps=save_images_every_n_steps,
            visualization_type=visualization_type,
            checkpoint_step=checkpoint_step,
        )

    def train_step(self, batch):
        pass

    def test_inputs(self, dataset):
        pass
