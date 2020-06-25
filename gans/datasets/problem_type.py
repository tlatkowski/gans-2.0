import enum


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
