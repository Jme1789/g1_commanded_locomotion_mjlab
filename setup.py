"""Installation script for the 'unitree_rl_mjlab' python package."""

from setuptools import find_packages, setup

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "mjlab==1.2.0",
    "mujoco-warp==3.5.0",
    "fastapi==0.139.2",
    "uvicorn==0.51.0",
    "httpx==0.28.1",
    "PyYAML==6.0.3",
]

# Installation operation
setup(
    name="unitree_rl_mjlab",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "src.gamepad_calibrator": ["static/*"],
    },
    version="0.0.1",
    install_requires=INSTALL_REQUIRES,
)
