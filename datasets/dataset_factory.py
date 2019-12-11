import enum

from datasets import cifar10
from datasets import fashion_mnist
from datasets import mnist
from datasets import summer2winter

class ProblemType(enum.Enum):
    VANILLA_MNIST = 0,
    VANILLA_FASHION_MNIST = 1
    VANILLA_CIFAR10 = 2
    CONDITIONAL_MNIST = 3
    CONDITIONAL_FASHION_MNIST = 4
    CONDITIONAL_CIFAR10 = 5
    CYCLE_SUMMER2WINTER = 6


def dataset_type_values():
    return [i.name for i in ProblemType]


def get_dataset(input_params, dataset_type: ProblemType):
    if dataset_type == ProblemType.VANILLA_MNIST.name:
        return mnist.MnistDataset(input_params)
    elif dataset_type == ProblemType.VANILLA_FASHION_MNIST.name:
        return fashion_mnist.FashionMnistDataset(input_params)
    elif dataset_type == ProblemType.VANILLA_CIFAR10.name:
        return cifar10.Cifar10Dataset(input_params)
    elif dataset_type == ProblemType.CONDITIONAL_MNIST.name:
        return mnist.MnistDataset(input_params, with_labels=True)
    elif dataset_type == ProblemType.CONDITIONAL_FASHION_MNIST.name:
        return fashion_mnist.FashionMnistDataset(input_params, with_labels=True)
    elif dataset_type == ProblemType.CONDITIONAL_CIFAR10.name:
        return cifar10.Cifar10Dataset(input_params, with_labels=True)
    elif dataset_type == ProblemType.CYCLE_SUMMER2WINTER.name:
        return summer2winter.SummerToWinterDataset()
    else:
        raise NotImplementedError
