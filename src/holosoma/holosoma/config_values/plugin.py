"""Registered plugin presets, selectable as ``plugin.<key>:<variant>`` on the CLI."""

from holosoma.config_types.plugin import (
    ClockPublishPluginConfig,
    GantryControlPluginConfig,
    NoOpPluginConfig,
    PluginConfig,
    ROS2OdometryPluginConfig,
)
from holosoma.utils.config_registry import ConfigRegistry, deprecated_defaults_alias

PLUGIN_REGISTRY = ConfigRegistry(PluginConfig, group="holosoma.config.plugin")

# `none` disables a slot (plugin.<key>:none), mirroring every other config family's `none`
# preset. Unlike the scalar families (which register a literal None), this dict field
# registers a real no-op config so the field type stays uniform. Extensions register their own
# via the entry-point group above or a --import-file that calls PLUGIN_REGISTRY.add(...).
none = PLUGIN_REGISTRY.add("none", NoOpPluginConfig())

# ROS2 example presets. Their impls import rclpy (optional dep: holosoma[ros2]); the
# configs stay rclpy-free, so registering them here does not require ROS.
clock_publish = PLUGIN_REGISTRY.add("clock_publish", ClockPublishPluginConfig())
gantry_control = PLUGIN_REGISTRY.add("gantry_control", GantryControlPluginConfig())

# Robot base pose/velocity as nav_msgs/Odometry — a self-sourced (non-camera) egress plugin that
# reads robot_root_states each control step. Rides the same in-process rclpy transport as the image
# egress (no CycloneDDS entanglement with the Unitree SDK bridge).
odometry = PLUGIN_REGISTRY.add("odometry", ROS2OdometryPluginConfig())

__getattr__ = deprecated_defaults_alias(__name__, PLUGIN_REGISTRY)
