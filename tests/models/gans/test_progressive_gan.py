import tensorflow as tf

from gans.models.gans import progressive_gan as pgan
from gans.models.generators.image_to_image import resnets


class TestProgressiveGANs(tf.test.TestCase):

    def test_build_progressive_generators(self):
        generators = resnets.build_progressive_generators(
            start_dim=(4, 4, 3),
            num_scales=4,
            r=2,
        )
        progressive_gan = pgan.ProgressiveGAN(
            generators=generators,
        )
        z = [tf.ones(shape=[4, 4, 4, 3]),
             tf.ones(shape=[4, 8, 8, 3]),
             tf.ones(shape=[4, 16, 16, 3]),
             tf.ones(shape=[4, 32, 32, 3])]
        outputs = progressive_gan.predict(z)
        raise NotImplementedError


    def test_build_discriminators(self):
        generators = resnets.build_patch_discriminators(
            start_dim=(4, 4, 3),
            num_scales=4,
            r=2,
        )
        progressive_gan = pgan.ProgressiveGAN(
            generators=generators,
        )
        z = [tf.ones(shape=[4, 4, 4, 3]),
             tf.ones(shape=[4, 8, 8, 3]),
             tf.ones(shape=[4, 16, 16, 3]),
             tf.ones(shape=[4, 32, 32, 3])]
        outputs = progressive_gan.predict(z)
        raise NotImplementedError