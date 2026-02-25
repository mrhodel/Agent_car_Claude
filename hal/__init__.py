"""
hal/__init__.py
Hardware Abstraction Layer — top-level convenience re-exports.
"""
from .yahboom_board import YahboomBoard
from .motors import MotorController
from .ultrasonic import UltrasonicSensor
from .gimbal import Gimbal
from .camera import Camera
from .stream_server import MJPEGStreamer

__all__ = ["YahboomBoard", "MotorController", "UltrasonicSensor", "Gimbal", "Camera",
           "MJPEGStreamer"]
