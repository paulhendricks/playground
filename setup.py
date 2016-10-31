from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='playground',
    version='0.1.0',
    description='A playground for evaluating Reinforcement Learning algorithms.',
    long_description=long_description,
    url='https://github.com/paulhendricks/playground',
    author='Paul Hendricks',
    author_email='paul.hendricks.2013@owu.edu',
    license='MIT',
    packages=find_packages(exclude=['examples', 'scripts']),
)
