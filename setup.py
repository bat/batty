from setuptools import setup, find_packages
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()

setup(
    name='batty',
    version='0.0.4',
    packages=find_packages(),
    license='Apache 2.0',
    author='Philipp Eller',
    author_email='peller.phys@gmail.com',
    url='https://github.com/philippeller/batty',
    description='BAT to Python',
    long_description=long_description,
    long_description_content_type='text/markdown',

    python_requires='>=3.6',

    install_requires=[
        'numpy',
        'julia',
        'uncertainties',
        'awkward',
    ],
)
