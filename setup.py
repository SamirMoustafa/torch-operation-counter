from setuptools import find_packages, setup

__version__ = "0.1"

install_requires = [
    "torch>=1.13.1",
]

pkg_name = "torch_ops_counter"
setup(
    name=pkg_name,
    version=__version__,
    install_requires=install_requires,
    packages=find_packages(where=pkg_name),
    author="Samir Moustafa",
    author_email="samir.moustafa.97@gmail.com",
    url="https://gitlab.cs.univie.ac.at/samirm97cs/torch_ops_counter",
)