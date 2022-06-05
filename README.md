# cbf_utils
Python package for using simple control barrier function.

## Installation
```sh
cd cbf_utils
python3 -m pip install -I -r requirements.txt
```

## example
- field_cbf_qp_optimizer.py

<img src=asset/field_cbf.gif>

## Use in ROS package
```sh
cd ~/catkin_ws/src/package_name
mkdir src && cd src
git submodule add -f https://github.com/toshi67026/cbf_utils.git
```

Also, edit setup.py as follows.
```py
from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(packages=["cbf_utils"], package_dir={"": "src"})

setup(**setup_args)
```

## tools
- mypy
```sh
./tools/run_mypy.sh
```
- black
```sh
./tools/run_black.sh
```
