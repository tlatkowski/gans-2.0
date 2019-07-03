import tensorflow as tf
from tensorflow.python.keras import Input

from models import attention


class TestModels(tf.test.TestCase):
    
    def testMultiHeadAttentionModelOutputShape(self):
        max_sequence_length = 50
        embedding_size = 64
        inputs = Input(shape=[max_sequence_length, embedding_size])
        g = attention.multihead_attention_model(inputs)
        actual_output_shape = g.shape
        expected_output_shape = (1, 28, 28, 1)
        self.assertEqual(actual_output_shape, expected_output_shape)
