import tensorflow as tf
from easydict import EasyDict as edict

from data_loaders import cifar10
from data_loaders import mnist


class TestDataLoaders(tf.test.TestCase):
    
    def testMnistDataLoader(self):
        input_params = edict({
            'batch_size': 4,
            'buffer_size': 60000
        })
        mnist.load_data(input_params)
    
    def testFashionMnistDataLoader(self):
        raise NotImplementedError
    
    def testCifar10DataLoader(self):
        input_params = edict({'batch_size': 4, 'buffer_size': 60000})
        dataset = cifar10.load_data(input_params)
        print('pass')
