import tensorflow as tf
from easydict import EasyDict as edict

from gans.models.discriminators import discriminator


class TestDiscriminators(tf.test.TestCase):

    def test_discriminator_output_shape(self):
        model_parameters = edict({
            'img_height':   32,
            'img_width':    32,
            'num_channels': 3,
        })
        d = discriminator.Discriminator(model_parameters)
        inputs = tf.ones(shape=[4, 32, 32, 3])
        output_img = d(inputs)

        actual_shape = output_img.shape
        expected_shape = (4, 1)
        self.assertEqual(actual_shape, expected_shape)
