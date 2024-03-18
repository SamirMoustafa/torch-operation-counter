from setuptools import find_packages, setup

__version__ = "0.3"

install_requires = [
    "torch>=1.13.1",
]

pkg_name = "torch_operation_counter"
setup(
    name=pkg_name,
    version=__version__,
    install_requires=install_requires,
    packages=find_packages(),
    author="Samir Moustafa",
    author_email="samir.moustafa.97@gmail.com",
    url="https://github.com/SamirMoustafa/torch-operation-counter/",
)
