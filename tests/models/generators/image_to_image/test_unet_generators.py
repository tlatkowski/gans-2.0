import tensorflow as tf
from easydict import EasyDict as edict

from gans.models.generators.image_to_image import unet


class TestUNetGenerators(tf.test.TestCase):

    def test_unet_generator_output_shape(self):
        model_parameters = edict({
            'latent_size':  100,
            'img_height':   256,
            'img_width':    256,
            'num_channels': 3,

        })
        g = unet.UNetGenerator(model_parameters)
        z = tf.ones(shape=[4, 256, 256, 3])
        output_img = g(z)

        actual_shape = output_img.shape
        expected_shape = (4, 256, 256, 3)
        self.assertEqual(actual_shape, expected_shape)

    def test_unet_subpixel_generator_output_shape(self):
        model_parameters = edict({
            'latent_size':  100,
            'img_height':   256,
            'img_width':    256,
            'num_channels': 3,

        })
        g = unet.UNetSubpixelGenerator(model_parameters)
        z = tf.ones(shape=[4, 256, 256, 3])
        output_img = g(z)

        actual_shape = output_img.shape
        expected_shape = (4, 256, 256, 3)
        self.assertEqual(actual_shape, expected_shape)
