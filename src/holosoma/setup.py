from setuptools import setup

# Robot SDKs are published to PyPI (FAR forks). The import names are unchanged
# (`unitree_interface`, `booster_robotics_sdk`) — only the distribution names
# differ from the historical GitHub-release wheels.
UNITREE_VERSION = "0.1.5"
BOOSTER_VERSION = "0.1.1"

setup(
    extras_require={
        "unitree": [f"far-unitree-sdk=={UNITREE_VERSION}"],
        "booster": [f"far-booster-sdk=={BOOSTER_VERSION}"],
        # ROS2 example plugins (clock_publish, gantry_control). rclpy and the ROS message
        # packages ship with a ROS2 distro / RoboStack, not PyPI, so this extra is a
        # marker/opt-in rather than a pip-installable set — install rclpy from your ROS
        # environment. The plugins import rclpy lazily, so core stays ROS-free without it.
        "ros2": [],
    },
    # Entry points are declared in pyproject.toml [project.entry-points.*]
)
