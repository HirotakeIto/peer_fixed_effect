# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages


with open('README.rst', encoding="utf-8") as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='peer_fixed_effect',
    version='0.1.0',
    description='python package of Arcidiacono et al. (2007) ',
    long_description=readme,
    author='Hirotake Ito',
    author_email='itouhrtk@gmail.com',
    install_requires=['numpy', 'pandas', 'sklearn'],
    dependency_links=[],
    url='https://github.com/kennethreitz/samplemod',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

