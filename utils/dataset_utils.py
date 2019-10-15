import enum

from data_loaders import cifar10
from data_loaders import fashion_mnist
from data_loaders import mnist


class ProblemType(enum.Enum):
    VANILLA_MNIST = 0,
    VANILLA_FASHION_MNIST = 1
    VANILLA_CIFAR10 = 2
    CONDITIONAL_MNIST = 3
    CONDITIONAL_FASHION_MNIST = 4
    CONDITIONAL_CIFAR10 = 5


def dataset_type_values():
    return [i.name for i in ProblemType]


def problem_factory(input_params, dataset_type: ProblemType):
    if dataset_type == ProblemType.VANILLA_MNIST.name:
        return mnist.load_data(input_params)
    elif dataset_type == ProblemType.VANILLA_FASHION_MNIST.name:
        return fashion_mnist.load_data(input_params)
    elif dataset_type == ProblemType.VANILLA_CIFAR10.name:
        return cifar10.Cifar10Dataset(input_params)
    elif dataset_type == ProblemType.CONDITIONAL_MNIST.name:
        return mnist.load_data_with_labels(input_params)
    elif dataset_type == ProblemType.CONDITIONAL_FASHION_MNIST.name:
        return fashion_mnist.load_data_with_labels(input_params)
    elif dataset_type == ProblemType.CONDITIONAL_CIFAR10.name:
        return cifar10.Cifar10Dataset(input_params, with_labels=True)
    else:
        raise NotImplementedError
