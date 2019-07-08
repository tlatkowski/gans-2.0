import tensorflow as tf

from models import txt2img_gan


class TestModels(tf.test.TestCase):
    
    def testText2ImageGAN(self):
        model = txt2img_gan.Text2ImageGAN()
        model.fit()
        raise NotImplementedError
