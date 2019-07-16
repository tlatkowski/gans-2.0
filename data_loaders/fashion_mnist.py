from tensorflow.python.keras import datasets


def load_data():
    fashion_mnist = datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    return (train_images, train_labels), (test_images, test_labels)
