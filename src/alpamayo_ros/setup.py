import os
from glob import glob

from setuptools import find_packages, setup

package_name = "alpamayo_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            os.path.join("share", package_name, "launch"),
            glob(os.path.join("launch", "*.launch.py")),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="shintarosakoda",
    maintainer_email="shintaro.sakoda@tier4.jp",
    description="ROS 2 interface for Alpamayo inference.",
    license="Apache License 2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "alpamayo_node = alpamayo_ros.alpamayo_node:main",
            "dp_stack_trajectory_porter = alpamayo_ros.dp_stack_trajectory_porter:main",
        ],
    },
)
