import tensorflow as tf
from tensorflow.python.keras import Input

from data_loaders import coco


class TestDataLoaders(tf.test.TestCase):
    
    def testCocoDataLoader(self):
        images_paths, captions_texts = coco.load_data()
        print(images_paths)
