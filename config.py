import os

home = r'D:\Datasets'
root_dirs = {
    'bird': home + r'\CUB_200_2011',
    'car':  home + '/Hello/Data/car',
    'air':  home + '/Hello/Data/aircraft',
    'dog':  home + '/Hello/Data/dog'
}

class_nums = {
    'bird': 200,
    'car': 196,
    'air': 100,
    'dog': 120
}

HyperParams = {
    'alpha': 0.5,
    'beta':  0.5,
    'gamma': 1,
    'kind': 'bird',
    'bs': 64,
    'epoch': 200,
    'arch': 'resnet50'
}
