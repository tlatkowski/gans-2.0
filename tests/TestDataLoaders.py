import tensorflow as tf
from easydict import EasyDict as edict

from data_loaders import cifar10


class TestDataLoaders(tf.test.TestCase):
    
    def testMnistDataLoader(self):
        raise NotImplementedError
    
    def testFashionMnistDataLoader(self):
        raise NotImplementedError
    
    def testCifar10DataLoader(self):
        input_params = edict({'batch_size': 4, 'buffer_size': 60000})
        dataset = cifar10.load_data(input_params)
        print('pass')
