import tensorflow as tf
from easydict import EasyDict as edict

from models.generators import conditional_random_to_image
from models.generators import random_to_image


class TestModels(tf.test.TestCase):
    
    def testRandomToImageGeneratorOutputShape(self):
        hidden_size = 100
        g = random_to_image.RandomToImageGenerator(hidden_size)
        z = tf.random_normal(shape=[1, 100])
        output_img = g(z)
        expected_shape = (1, 28, 28, 1)
        self.assertEqual(output_img.shape, expected_shape)
    
    def testRandomToImageConditionalGeneratorOutputShape(self):
        input_params = edict({
            'hidden_size': 100,
            'num_classes': 10
        })
        g = conditional_random_to_image.RandomToImageConditionalGenerator(input_params)
        z = tf.random.normal(shape=[1, 100])
        class_id = tf.one_hot(indices=[1], depth=10)
        output_img = g([z, class_id])
        expected_shape = (1, 28, 28, 1)
        self.assertEqual(output_img.shape, expected_shape)
