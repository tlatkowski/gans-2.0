import tensorflow as tf
from easydict import EasyDict as edict

from gans.models.generators.latent_to_image import latent_to_image


class TestLatentToImageGenerators(tf.test.TestCase):

    def test_random_to_image_generator_output_shape(self):
        input_params = edict({
            'latent_size': 100
        })
        g = latent_to_image.LatentToImageGenerator(input_params)
        z = tf.random.normal(shape=[1, 100])
        output_img = g(z)
        expected_shape = (1, 28, 28, 1)
        self.assertEqual(output_img.shape, expected_shape)
