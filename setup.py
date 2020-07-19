from setuptools import setup, find_packages

import os

setup(
    name='Interactive Plotting',
    version='0.0.5',
    description='Interactive plotting functions for scanpy',
    url='https://github.com/theislab/interactive_plotting',
    license='MIT',
    packages=find_packages(),
    setup_requires=['setuptools_scm'],
    include_package_data=True,
    install_requires=list(map(str.strip,
                              open(os.path.abspath('requirements.txt'), 'r').read().split())),

    zip_safe=False
)
