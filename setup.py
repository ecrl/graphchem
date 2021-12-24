from setuptools import find_packages, setup

setup(
    name='graphchem',
    version='1.2.0',
    description='Graph-based machine learning for chemical property prediction',
    url='https://github.com/ecrl/graphchem',
    author='Travis Kessler',
    author_email='Travis_Kessler@student.uml.edu',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'torch>=1.8.2',
        'pytorch-nlp>=0.5.0',
        'torch-scatter>=2.0.7',
        'torch-sparse>=0.6.10',
        'torch-geometric>=1.7.2'
    ],
    package_data={
        'graphchem': [
            'datasets/static/*'
        ]
    },
    zip_safe=False
)
