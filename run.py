import argparse


def run_experiment(dataset, gan_type):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset',
                        required=True,
                        help='The path to training images')
    
    parser.add_argument('--gan_type',
                        required=True,
                        help='The path to training images')
    
    args = parser.parse_args()
    
    run_experiment(argparse.dataset, argparse.gan_type)
