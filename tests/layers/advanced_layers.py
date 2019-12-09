import tensorflow as tf

from layers import advanced_layers


class TestAdvancedLayers(tf.test.TestCase):
    
    def test_densely_connected_residual_block_shape(self):
        inputs = tf.ones(shape=(4, 32, 32, 64))
        outputs = advanced_layers.densely_connected_residual_block(inputs)
        expected_shape = (4, 32, 32, 64)
        self.assertEqual(outputs.shape, expected_shape)
