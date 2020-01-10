from setuptools import find_packages, setup

setup(
    name='graphchem',
    version='0.3.0',
    description='Graph representations of molecules derived from SMILES'
    ' strings',
    url='http://github.com/tjkessler/graphchem',
    author='Travis Kessler',
    author_email='travis.j.kessler@gmail.com',
    license='MIT',
    packages=find_packages(),
    install_requires=['torch==1.3.1'],
    zip_safe=False
)
