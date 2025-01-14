from setuptools import find_packages, setup

setup(
    name='flow-simulation-pinn',
    version='0.1.0',
    description='',
    author='Timur Lidzhiev',
    author_email='trlidzhiev@gmail.com',
    packages=find_packages(where='pinns'),
    install_requires=[
        'numba==0.60.0',
        'numpy==2.0.0',
        'torch==2.5.1',
        'scipy==1.14.1',
        'matplotlib==3.10.0',
        'hydra-core==1.3.2',
        'wandb==0.19.1',
        'tqdm==4.67.1',
        'optuna==4.1.0',
        'ipykernel==6.29.5',
        'plotly==5.24.1',
    ],
    extras_require={
        'dev': [
            'pytest==8.3.3',
            'torchviz==0.0.2',
        ]
    },
    python_requires='>=3.10.12',
    zip_safe=False,
)
