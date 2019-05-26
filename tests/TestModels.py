import tensorflow as tf

from models import generator


class TestModels(tf.test.TestCase):
    
    def testGenerator(self):
        hidden_size = 100
        g = generator.Generator(hidden_size)
        z = tf.random_normal(shape=[1, 100])
        output_img = g(z)
        expected_shape = (1, 28, 28, 1)
        self.assertEqual(output_img.shape, expected_shape)
