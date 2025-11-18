"""
Setup configuration for sleep_staging_models package
"""
from setuptools import setup

setup(
    name='sleep_staging_models',
    version='0.1.0',
    description='PPG-based sleep staging models with cross-attention',
    author='James',
    packages=['.'],
    package_dir={'': '.'},
    python_requires='>=3.11',
    install_requires=[
        'torch>=2.0.0',
        'numpy>=1.24.0',
        'h5py>=3.8.0',
        'scipy>=1.10.0',
        'matplotlib>=3.7.0',
        'seaborn>=0.12.0',
        'tensorboard>=2.12.0',
        'pyyaml>=6.0',
        'tqdm>=4.65.0',
        'scikit-learn>=1.2.0',
    ],
    extras_require={
        'dev': [
            'jupyter>=1.0.0',
            'ipykernel>=6.22.0',
            'pandas>=2.0.0',
            'python-dotenv>=1.0.0',
        ]
    },
)
