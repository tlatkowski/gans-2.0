import tensorflow as tf
from easydict import EasyDict as edict

from gans.models.discriminators import conditional_discriminator


class TestConditionalDiscriminators(tf.test.TestCase):

    def test_conditional_discriminator_output_shape(self):
        model_parameters = edict({
            'img_height':   32,
            'img_width':    32,
            'num_channels': 3,
        })
        d = conditional_discriminator.ConditionalDiscriminator(model_parameters)
        inputs = tf.ones(shape=[4, 32, 32, 3])
        class_id = tf.zeros(shape=(4,))
        output_img = d([inputs, class_id])

        actual_shape = output_img.shape
        expected_shape = (4, 1)
        self.assertEqual(actual_shape, expected_shape)

    def test_conditional_discriminator_cifar10_output_shape(self):
        model_parameters = edict({
            'img_height':   32,
            'img_width':    32,
            'num_channels': 3,
        })
        d = conditional_discriminator.ConditionalDiscriminatorCifar10(model_parameters)
        inputs = tf.ones(shape=[4, 32, 32, 3])
        class_id = tf.zeros(shape=(4,))
        output_img = d([inputs, class_id])

        actual_shape = output_img.shape
        expected_shape = (4, 1)
        self.assertEqual(actual_shape, expected_shape)

