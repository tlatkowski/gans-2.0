from easydict import EasyDict as edict

from gans.datasets import summer2winter
from gans.models.generators.image_to_image import resnets
from gans.trainers import optimizers
from gans.trainers import progressive_gan_trainer

model_parameters = edict({
    'img_height':                  128,
    'img_width':                   128,
    'num_channels':                3,
    'batch_size':                  16,
    'num_epochs':                  10,
    'buffer_size':                 1000,
    'latent_size':                 100,
    'learning_rate_generator':     0.0001,
    'learning_rate_discriminator': 0.0001,
    'save_images_every_n_steps':   10
})
dataset = summer2winter.SummerToWinterDataset(model_parameters)

generators = resnets.build_progressive_generators(
    start_dim=(16, 16, 3),
    num_scales=4,
    r=2,
)

discriminators = resnets.build_patch_discriminators(
    start_dim=(16, 16, 3),
    num_scales=4,
    r=2,
)

generator_optimizer = optimizers.Adam(
    learning_rate=model_parameters.learning_rate_generator,
    beta_1=0.5,
)

discriminator_optimizer = optimizers.Adam(
    learning_rate=model_parameters.learning_rate_discriminator,
    beta_1=0.5,
)

gan_trainer = progressive_gan_trainer.ProgressiveGANTrainer(
    batch_size=model_parameters.batch_size,
    generators=generators,
    discriminators=discriminators,
    training_name='PROGRESSIVE_GAN_ANIMATION',
    generators_optimizers=[generator_optimizer],
    discriminators_optimizers=[discriminator_optimizer],
)

gan_trainer.train(
    dataset=dataset,
    num_epochs=model_parameters.num_epochs,
)
