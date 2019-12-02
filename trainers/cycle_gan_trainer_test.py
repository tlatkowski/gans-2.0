from datasets import summer2winter
from models.discriminators import cycle_discriminator
from models.generators import image_to_image
from trainers import cycle_gan_trainer

generator = image_to_image.CycleGenerator()
discriminator = cycle_discriminator.Discriminator()
dataset = summer2winter.SummerToWinterDataset()
cycle_gan_trainer.CycleGANTrainer(
    batch_size=4,
    generator=generator,
    discriminator=discriminator,
    dataset_type='summer2winter',
    lr_generator=0.0005,
    lr_discriminator=0.0002,
    continue_training=False,
).train(dataset, 10)
