from os import path
from setuptools import find_packages, setup

__version__ = "0.3.7"

install_requires = [
    "torch>=1.13.1",
]

# Get the long description from the README file
with open(path.join(path.abspath(path.dirname(__file__)), "README.md"), encoding="utf-8") as f:
    long_description = f.read()

pkg_name = "torch_operation_counter"
setup(
    name=pkg_name,
    version=__version__,
    install_requires=install_requires,
    packages=find_packages(),
    author="Samir Moustafa",
    author_email="samir.moustafa.97@gmail.com",
    url="https://github.com/SamirMoustafa/torch-operation-counter/",
    long_description=long_description,
    long_description_content_type="text/markdown",
)
