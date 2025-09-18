from setuptools import setup, find_packages

setup(
    name='FEA',
    version='2.0.0',
    packages=find_packages(),
    install_requires=[
        'torch>=2.1.0',
    ],
    author='Song Zenan',
)