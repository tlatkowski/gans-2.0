import tensorflow as tf
from easydict import EasyDict as edict

from gans.models.generators.latent_to_image import conditional_latent_to_image


class TestConditionalLatentToImageGenerators(tf.test.TestCase):

    def test_conditional_random_to_image_generator_output_shape(self):
        input_params = edict({
            'latent_size': 100,
            'num_classes': 10
        })
        g = conditional_latent_to_image.LatentToImageConditionalGenerator(input_params)
        z = tf.random.normal(shape=(1, 100))
        class_id = tf.zeros(shape=(1,))
        output_img = g([z, class_id])
        expected_shape = (1, 28, 28, 1)
        self.assertEqual(output_img.shape, expected_shape)
