import numpy as np
import rerun as rr

import rclpy
from rclpy.node import Node
from vr_ros2_bridge_msgs.msg import ControllersInfo, ControllerInfo


class AxisVelocityHandler:
    """
    A class for handling the state of a controller axis, and converting it to a velocity.
    """

    def __init__(self, buffer_size=15):
        self.buffer_size = buffer_size
        self.buffer = []
        self.was_touching = False

    def update(self, current_value: float):
        is_touching = (current_value != 0.)  # exactly 0.0 means not touching. No human input will be exactly 0
        if is_touching:
            if not self.was_touching:
                self.was_touching = True
                self.buffer = []
        else:
            self.was_touching = False
            return 0

        self.buffer.append(current_value)
        if len(self.buffer) < self.buffer_size:
            return 0

        # estimate velocity, averaging over the last 10 samples
        array = np.array(self.buffer)
        deltas = array[1:] - array[:-1]
        velocity = np.mean(deltas)

        self.buffer.pop(0)

        return velocity


class TestAxisVelocityHandler(Node):

    def __init__(self):
        super().__init__("test_axis_velocity_handler")
        self.trackpad_y_axis_velocity_handler = AxisVelocityHandler()

        self.vr_sub = self.create_subscription(ControllersInfo, "vr_controller_info", self.on_controllers_info, 10)

    def on_controllers_info(self, msg: ControllersInfo):
        if len(msg.controllers_info) == 0:
            return

        controller_info: ControllerInfo = msg.controllers_info[0]

        trackpad_y_velocity = self.trackpad_y_axis_velocity_handler.update(controller_info.trackpad_axis_touch_y)
        rr.log('linear_velocity_scale', rr.TimeSeriesScalar(trackpad_y_velocity))
        rr.log('trackpad_y', rr.TimeSeriesScalar(controller_info.trackpad_axis_touch_y))


def main():
    rr.init("generate_data_from_vr")
    rr.connect()

    rclpy.init()
    node = TestAxisVelocityHandler()
    rclpy.spin(node)
