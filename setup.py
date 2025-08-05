from setuptools import setup, find_packages

setup(
    name='conforml',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'scikit-learn>=1.0.0',
        'statsmodels>=0.13.0',
        'matplotlib>=3.5.0',
        'torch>=1.10.0',
        'prophet>=1.1.0'
    ],
    entry_points={
        'console_scripts': [
            'conforml=conforml.cli:main',
        ],
    },
)