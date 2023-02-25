from setuptools import find_packages, setup


setup(
    name="CryoEnv",
    version='1.0.2',
    author='Felix Wagner',
    author_email="felix.wagner@oeaw.ac.at",
    description='CryoEnv - Reinforcement learning for cryogenic calorimeters.',
    url="https://github.com/fewagner/CryoEnv",
    license='GPLv3',
    packages=find_packages(include=['cryoenv', 'cryoenv.*']),
    install_requires=[
        # "gym==0.19.0",
        # "stable_baselines3==1.3.0",
        # "tqdm==4.62.3",
        # "numba",
        # "torch==1.12.1",
        # "numbalsoda==0.3.4",
        # "tensorboard==2.10.1",
    ],
    python_requires='>=3.8',
)
