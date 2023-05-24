#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

requirements = []

test_requirements = [ ]

setup(
    python_requires='>=3.6',
    install_requires=requirements,
    keywords='al_ntk',
    name='al_ntk',
    packages=find_packages(include=['al_ntk', 'al_ntk.*'])
)
