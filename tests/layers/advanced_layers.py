import tensorflow as tf

from gans.layers import advanced_layers


class TestAdvancedLayers(tf.test.TestCase):

    def test_densely_connected_residual_block_output_shape(self):
        inputs = tf.ones(shape=(4, 32, 32, 64))
        outputs = advanced_layers.densely_connected_residual_block(inputs)
        expected_shape = (4, 32, 32, 64)
        self.assertEqual(outputs.shape, expected_shape)

    def test_channel_attention_block_output_shape(self):
        inputs = tf.ones(shape=(4, 32, 32, 64))
        outputs = advanced_layers.channel_attention_block(inputs, r=2)
        expected_shape = (4, 32, 32, 64)
        self.assertEqual(outputs.shape, expected_shape)

    def test_subpixel_upsampling_layer_output_shape(self):
        inputs = tf.ones(shape=(4, 32, 32, 64))
        outputs = advanced_layers.subpixel_upsampling(inputs, r=2)
        expected_shape = (4, 64, 64, 16)
        self.assertEqual(outputs.shape, expected_shape)
