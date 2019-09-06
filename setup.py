from setuptools import setup

import os

setup(
    name='Interactive Plotting',
    version='0.0.2',
    description='Interactive plotting functions for scanpy',
    url='https://github.com/theislab/interactive_plotting',
    license='MIT',
    packages=['interactive_plotting'],
    install_requires=list(map(str.strip,
                              open(os.path.abspath('requirements.txt'), 'r').read().split())),

    zip_safe=False
)
