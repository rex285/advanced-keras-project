from setuptools import setup, find_packages

setup(
    name='keras_project',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'tensorflow>=2.8.0',
        'numpy',
        'pandas'
    ]
)