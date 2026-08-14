from setuptools import find_packages, setup

package_name = "autocar_watchdog"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Aiman Hazim",
    maintainer_email="dev@aimanhazim.com",
    description="Latency budgets; owns second_miss (PR-02).",
    license="MIT",
    entry_points={"console_scripts": ["watchdog_node = autocar_watchdog.watchdog_node:main"]},
)
