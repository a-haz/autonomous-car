from setuptools import find_packages, setup

package_name = "autocar_bringup"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "cyclone_localhost.xml"]),
        ("share/" + package_name + "/launch", ["launch/vehicle.launch.py"]),
        ("share/" + package_name + "/config", ["config/odd_bits.yaml", "config/topics.yaml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Aiman Hazim",
    maintainer_email="dev@aimanhazim.com",
    description="Launch graph for the autonomous-car cage.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [],
    },
)
