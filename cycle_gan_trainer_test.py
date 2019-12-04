from datasets import summer2winter
from models.discriminators import cycle_discriminator
from models.generators import u_net
# from models.generators import image_to_image
from trainers import cycle_gan_trainer

generator_a = u_net.UNetGenerator()
# generator_a = image_to_image.CycleGenerator()
# generator_b = image_to_image.CycleGenerator()
generator_b = u_net.UNetGenerator()
discriminator_a = cycle_discriminator.Discriminator()
discriminator_b = cycle_discriminator.Discriminator()
dataset = summer2winter.SummerToWinterDataset()
cycle_gan_trainer.CycleGANTrainer(
    batch_size=4,
    generator=[generator_a, generator_b],
    discriminator=[discriminator_a, discriminator_b],
    dataset_type='summer2winter',
    lr_generator=0.0005,
    lr_discriminator=0.0002,
    continue_training=False,
).train(dataset, 10)
