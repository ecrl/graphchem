from setuptools import find_packages, setup
import os


def get_readme():

    with open('README.md', 'r') as f:
        return f.read()


def get_version_info():

    version_path = os.path.join('graphchem', 'version.py')
    file_vars = {}
    with open(version_path, 'r') as f:
        exec(f.read(), file_vars)
    f.close()
    return file_vars['__version__']


setup(
    name='graphchem',
    version=get_version_info(),
    description='Graph-based models for chemical property prediction',
    url='https://github.com/ecrl/graphchem',
    author='Travis Kessler',
    author_email='Travis_Kessler@student.uml.edu',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'rdkit-pypi==2022.9.5',
        'torch==2.0.0',
        'torch-geometric==2.3.0'
    ],
    package_data={
        'graphchem': [
            'datasets/static/*'
        ]
    },
    zip_safe=False
)
