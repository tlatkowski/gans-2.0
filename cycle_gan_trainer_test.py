from datasets import summer2winter
from models.discriminators import patch_discriminator
from models.generators import dense_net
# from models.generators import image_to_image
from trainers import cycle_gan_trainer

generator_a = dense_net.DenseNetGenerator()
# generator_a = image_to_image.CycleGenerator()
# generator_b = image_to_image.CycleGenerator()
generator_b = dense_net.DenseNetGenerator()
discriminator_a = patch_discriminator.PatchDiscriminator()
discriminator_b = patch_discriminator.PatchDiscriminator()
dataset = summer2winter.SummerToWinterDataset()
cycle_gan_trainer.CycleGANTrainer(
    batch_size=4,
    generator=[generator_a, generator_b],
    discriminator=[discriminator_a, discriminator_b],
    dataset_type='summer2winter',
    lr_generator=0.0002,
    lr_discriminator=0.0002,
    continue_training=False,
).train(dataset, 20)
