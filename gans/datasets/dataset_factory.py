from gans.datasets import cifar10
from gans.datasets import fashion_mnist
from gans.datasets import mnist
from gans.datasets import problem_type
from gans.datasets import summer2winter


def get_dataset(input_params, dataset_type: problem_type.ProblemType):
    if dataset_type == problem_type.ProblemType.VANILLA_MNIST.name:
        return mnist.MnistDataset(input_params)
    elif dataset_type == problem_type.ProblemType.VANILLA_FASHION_MNIST.name:
        return fashion_mnist.FashionMnistDataset(input_params)
    elif dataset_type == problem_type.ProblemType.VANILLA_CIFAR10.name:
        return cifar10.Cifar10Dataset(input_params)
    elif dataset_type == problem_type.ProblemType.CONDITIONAL_MNIST.name:
        return mnist.MnistDataset(input_params, with_labels=True)
    elif dataset_type == problem_type.ProblemType.CONDITIONAL_FASHION_MNIST.name:
        return fashion_mnist.FashionMnistDataset(input_params, with_labels=True)
    elif dataset_type == problem_type.ProblemType.CONDITIONAL_CIFAR10.name:
        return cifar10.Cifar10Dataset(input_params, with_labels=True)
    elif dataset_type == problem_type.ProblemType.CYCLE_SUMMER2WINTER.name:
        return summer2winter.SummerToWinterDataset(input_params)
    else:
        raise NotImplementedError
