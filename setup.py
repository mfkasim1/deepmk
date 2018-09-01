import os
from setuptools import setup, find_packages
from deepmk.version import get_version

setup(
    name='deepmk',
    version=get_version(),
    description="M. Kasim's deep learning framework",
    url='https://github.com/mfkasim91/deepmk',
    author='mfkasim91',
    author_email='firman.kasim@gmail.com',
    license='MIT',
    packages=find_packages(),
    python_requires=">=2.7",
    install_requires=[
        "torch>=0.4.1",
        "torchvision>=0.2.1",
        "numpy>=1.14.3",
        "scipy>=1.1.0",
        "matplotlib>=2.2.2"
    ],
    # extras_require={
    #     "bayesopt": ["george>=0.3.0", "emcee>=2.2.1"],
    # },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Computer Science",
        "License :: OSI Approved :: MIT License",

        "Programming Language :: Python :: 2.7"
    ],
    keywords="artificial-intelligence machine-learning",
    zip_safe=False
)
