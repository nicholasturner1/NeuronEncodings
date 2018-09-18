#!/usr/bin/env python3

from setuptools import setup

setup(
    name='neuronencodings',
    version='0.0.1',
    description='',
    author='Nicholas Turner, Sven Dorkenwald',
    author_email='nturner@cs.princeton.edu, svenmd@princeton.edu',
    url='https://github.com/nicholasturner1/NeuronEncodings',
    packages=['neuronencodings',
              'neuronencodings.data',
              'neuronencodings.loss',
              'neuronencodings.models',
              ]
)
