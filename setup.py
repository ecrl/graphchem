from setuptools import find_packages, setup

setup(
    name='graphchem',
    version='1.0.0',
    description='Graph-based machine learning for chemical property prediction',
    url='https://github.com/tjkessler/graphchem',
    author='Travis Kessler',
    author_email='Travis_Kessler@student.uml.edu',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'torch==1.8.0',
        'pytorch-nlp==0.5.0',
        'torch-scatter==2.0.6',
        'torch-sparse==0.6.9',
        'torch-cluster==1.5.9',
        'torch-spline-conv==1.2.1',
        'torch-geometric==1.6.3'
    ],
    package_data={
        'graphchem': [
            'datasets/static/*'
        ]
    },
    zip_safe=False
)
