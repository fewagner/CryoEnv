from setuptools import setup

setup(
    name="cryoenv",
    version="1.0.1",
    packages=["cryoenv"],
    package_dir={"cryoenv": "./cryoenv"},
    python_requires=">=3.8,<3.10",
    install_requires=[
        "gym==0.19.0",
        "stable_baselines3==1.3.0",
        "tqdm==4.62.3",
        "numba==0.54.1",
        "torch==1.12.1",
        "numbalsoda==0.3.4",
    ],
)
