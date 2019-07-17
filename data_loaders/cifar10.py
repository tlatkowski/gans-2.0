from tensorflow.python.keras import datasets


def load_data():
    cifar10 = datasets.cifar10
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    return (train_images, train_labels), (test_images, test_labels)
