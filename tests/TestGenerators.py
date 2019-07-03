import matplotlib.pyplot as plt
import tensorflow as tf

from models import generators


class TestModels(tf.test.TestCase):
    
    def testRandomToImageGeneratorOutputShape(self):
        hidden_size = 100
        g = generators.RandomToImageGenerator(hidden_size)
        z = tf.random_normal(shape=[1, 100])
        output_img = g(z)
        expected_shape = (1, 28, 28, 1)
        self.assertEqual(output_img.shape, expected_shape)
    
    def testTextToImageGeneratorOutputShape(self):
        g = generators.TextToImageGenerator(max_sequence_length=50, embedding_size=64)
        z = tf.random_normal(shape=[1, 100])
        output_img = g(z)
        expected_shape = (1, 28, 28, 1)
        self.assertEqual(output_img.shape, expected_shape)
    
    def testPlotGeneratorOutput(self):
        with self.session() as session:
            hidden_size = 100
            g = generators.RandomToImageGenerator(hidden_size)
            z = tf.random_normal(shape=[1, 100])
            session.run(tf.initialize_all_variables())
            generated_image = g(z)
            generated_image = session.run(generated_image)
            plt.imshow(generated_image[0, :, :, 0], cmap='gray')
