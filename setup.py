from os import path
from setuptools import setup, find_packages

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="batty",
    version="0.1.0",
    packages=find_packages(),
    license="MIT",
    author="Philipp Eller",
    author_email="peller.phys@gmail.com",
    url="https://github.com/philippeller/batty",
    description="BAT to Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "juliacall",
        "uncertainties",
        "awkward",
        "pygtc",
        "tqdm",
        "corner",
    ],
    package_data={'batty': ['batty/pybat.jl']}
)
