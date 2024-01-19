#!/usr/bin/env python3
"""
This script receives teleop commands from the Unity VR system, and sends them to Conq.
Press and hold the grip button on the VR controller to start recording an episode.
Release the grip button to stop recording an episode.
Press the trackpad button to stop the script.
Use the trigger to open and close the gripper, it's mapped to the open fraction of the gripper.
The motion will be relative in gripper frame to the pose of the gripper when you started recording.
see the vr_ros2_bridge repo for setup instructions.
"""

import time
from pathlib import Path

import bosdyn.client
import bosdyn.client.util
import numpy as np
import rerun as rr
from bosdyn.client.frame_helpers import get_a_tform_b, HAND_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME
from bosdyn.client.image import ImageClient
from bosdyn.client.lease import LeaseKeepAlive, LeaseClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.math_helpers import SE3Pose, Quat
from bosdyn.client.ray_cast import RayCastClient
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient
from bosdyn.client.robot_state import RobotStateClient

import rclpy
from conq.clients import Clients
from conq.data_recorder import ConqDataRecorder
from conq.hand_motion import hand_pose_cmd
from conq.manipulation import blocking_arm_command, add_follow_with_body
from conq.manipulation import open_gripper
from conq.utils import setup_and_stand, setup
from geometry_msgs.msg import Pose
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from turtlesim.msg import Pose
from vr.constants import VR_FRAME_NAME, ARM_POSE_CMD_DURATION, ARM_POSE_CMD_PERIOD
from vr.controller_utils import AxisVelocityHandler
from vr_ros2_bridge_msgs.msg import ControllersInfo, ControllerInfo


def controller_info_to_se3_pose(controller_info: ControllerInfo):
    pose_msg: Pose = controller_info.controller_pose
    return SE3Pose(x=pose_msg.position.x,
                   y=pose_msg.position.y,
                   z=pose_msg.position.z,
                   rot=Quat(w=pose_msg.orientation.w,
                            x=pose_msg.orientation.x,
                            y=pose_msg.orientation.y,
                            z=pose_msg.orientation.z))


def ease_func(speed):
    return max(speed, speed ** 2)


def np_sigmoid(x):
    return 1 / (1 + np.exp(-x))


class GenerateDataVRNode(Node):

    def __init__(self, conq_clients: Clients, follow_arm_with_body=True, viz_only=False):
        super().__init__("generate_data_from_vr")
        self.latest_action = None
        self.trackpad_y_axis_velocity_handler = AxisVelocityHandler()
        self.follow_arm_with_body = follow_arm_with_body
        self.conq_clients = conq_clients  # Node already has a clients attribute, hence the name change
        self.viz_only = viz_only
        self.hand_in_body0 = SE3Pose.from_identity()
        self.controller_in_vr0 = SE3Pose.from_identity()

        now = int(time.time())
        root = Path(f"data/conq_vr_data_{now}")
        self.recorder = ConqDataRecorder(root, conq_clients.state, conq_clients.image,
                                         sources=[
                                             'hand_color_image',
                                             'frontleft_fisheye_image',
                                             'frontright_fisheye_image',
                                         ],
                                         get_latest_action=self.get_latest_action,
                                         period=ARM_POSE_CMD_PERIOD)

        self.tf_broadcaster = TransformBroadcaster(self)
        self.vr_sub = self.create_subscription(ControllersInfo, "vr_controller_info", self.on_controllers_info, 10)

        self.has_started = False
        self.is_recording = False
        self.is_done = False

        self.on_reset()

    def send_cmd(self, target_hand_in_body: SE3Pose, open_fraction: float):
        hand_pose_msg = target_hand_in_body.to_proto()
        arm_cmd = RobotCommandBuilder.arm_pose_command_from_pose(hand_pose_msg, GRAV_ALIGNED_BODY_FRAME_NAME,
                                                                 seconds=ARM_POSE_CMD_DURATION)
        if self.follow_arm_with_body:
            arm_body_cmd = add_follow_with_body(arm_cmd)
        else:
            arm_body_cmd = arm_cmd

        gripper_cmd = RobotCommandBuilder.claw_gripper_open_fraction_command(open_fraction)
        arm_body_gripper_cmd = RobotCommandBuilder.build_synchro_command(arm_body_cmd, gripper_cmd)

        self.conq_clients.command.robot_command(arm_body_gripper_cmd)

    def on_done(self):
        if not self.viz_only:
            self.recorder.stop()

        open_gripper(self.conq_clients)
        blocking_arm_command(self.conq_clients, RobotCommandBuilder.arm_stow_command())

        raise SystemExit("Done!")

    def on_controllers_info(self, msg: ControllersInfo):
        if len(msg.controllers_info) == 0:
            return

        controller_info: ControllerInfo = msg.controllers_info[0]

        # set the flags for recording and done
        if not self.is_recording and controller_info.grip_button:
            self.is_recording = True
            self.has_started = True
            self.on_start_recording(controller_info)
        elif self.is_recording and not controller_info.grip_button:
            self.is_recording = False
            self.on_stop_recording()

        if self.has_started and controller_info.menu_button:
            self.is_done = True
            self.on_done()

        if not self.is_recording and controller_info.trackpad_button:
            self.on_reset()

        if self.is_recording:
            # for debugging purposes, we publish a fixed transform from vr frame to body frame,
            # even though we don't ever actually use this transform for controlling the robot.
            vr_to_body = TransformStamped()
            vr_to_body.header.stamp = self.get_clock().now().to_msg()
            vr_to_body.header.frame_id = VR_FRAME_NAME
            vr_to_body.child_frame_id = GRAV_ALIGNED_BODY_FRAME_NAME
            vr_to_body.transform.translation.x = 1.5
            vr_to_body.transform.rotation.w = 1.
            self.tf_broadcaster.sendTransform(vr_to_body)

            target_hand_in_body = self.get_target_in_body(controller_info)

            snapshot = self.conq_clients.state.get_robot_state().kinematic_state.transforms_snapshot
            hand_in_body = get_a_tform_b(snapshot, GRAV_ALIGNED_BODY_FRAME_NAME, HAND_FRAME_NAME)

            open_fraction = 1 - controller_info.trigger_axis

            # Save for the data recorder
            self.latest_action = {
                'target_hand_in_body': target_hand_in_body,
                'open_fraction': open_fraction
            }

            self.pub_se3_pose_to_tf(self.controller_in_vr0, 'controller_in_vr0', VR_FRAME_NAME)
            self.pub_se3_pose_to_tf(self.hand_in_body0, 'hand_in_body0', GRAV_ALIGNED_BODY_FRAME_NAME)
            self.pub_se3_pose_to_tf(hand_in_body, 'hand_in_body', GRAV_ALIGNED_BODY_FRAME_NAME)
            self.pub_se3_pose_to_tf(target_hand_in_body, 'target_in_body', GRAV_ALIGNED_BODY_FRAME_NAME)

            if not self.viz_only:
                self.send_cmd(target_hand_in_body, open_fraction)

    def get_velocity(self, controller_info: ControllerInfo):
        controller_info.controller_velocity

    def get_target_in_body(self, controller_info: ControllerInfo) -> SE3Pose:
        current_controller_in_vr = controller_info_to_se3_pose(controller_info)
        self.pub_se3_pose_to_tf(current_controller_in_vr, 'current VR', VR_FRAME_NAME)
        delta_in_vr = self.controller_in_vr0.inverse() * current_controller_in_vr

        # TODO: add a rotation here to account for the fact that we might not want VR controller frame and
        #  hand frame to be the same orientation?
        delta_in_body = delta_in_vr

        target_hand_in_body = self.hand_in_body0 * delta_in_body
        return target_hand_in_body

    def on_reset(self):
        if self.viz_only:
            return

        open_gripper(self.conq_clients)
        look_cmd = hand_pose_cmd(self.conq_clients, 0.8, 0, 0.2, 0, np.deg2rad(0), 0, duration=0.5)
        blocking_arm_command(self.conq_clients, look_cmd)

    def on_start_recording(self, controller_info: ControllerInfo):
        # Store the initial pose of the hand in body frame, as well as the controller pose which is in VR frame
        print(f"Starting recording episode {self.recorder.episode_idx}")

        self.controller_in_vr0 = controller_info_to_se3_pose(controller_info)
        snapshot = self.conq_clients.state.get_robot_state().kinematic_state.transforms_snapshot
        self.hand_in_body0 = get_a_tform_b(snapshot, GRAV_ALIGNED_BODY_FRAME_NAME, HAND_FRAME_NAME)

        mode = "unsorted"
        if not self.viz_only:
            self.recorder.start_episode(mode, "grasp hose")

    def on_stop_recording(self):
        print(f"Stopping recording episode {self.recorder.episode_idx}")
        if not self.viz_only:
            self.recorder.next_episode()

    def pub_se3_pose_to_tf(self, pose: SE3Pose, child_frame_name: str, parent_frame_name: str):
        t = TransformStamped()

        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = parent_frame_name
        t.child_frame_id = child_frame_name

        t.transform.translation.x = float(pose.x)
        t.transform.translation.y = float(pose.y)
        t.transform.translation.z = float(pose.z)

        t.transform.rotation.x = float(pose.rot.x)
        t.transform.rotation.y = float(pose.rot.y)
        t.transform.rotation.z = float(pose.rot.z)
        t.transform.rotation.w = float(pose.rot.w)

        self.tf_broadcaster.sendTransform(t)

    def get_latest_action(self, _):
        return self.latest_action


def main():
    np.seterr(all='raise')
    np.set_printoptions(precision=3, suppress=True)

    rr.init("generate_data_from_vr")
    rr.connect()

    # Creates client, robot, and authenticates, and time syncs
    viz_only = False
    sdk = bosdyn.client.create_standard_sdk('generate_dataset')
    # robot = sdk.create_robot('192.168.80.3')
    robot = sdk.create_robot('10.0.0.3')
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()

    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    manipulation_api_client = robot.ensure_client(ManipulationApiClient.default_service_name)
    image_client = robot.ensure_client(ImageClient.default_service_name)
    rc_client = robot.ensure_client(RayCastClient.default_service_name)

    lease_client.take()

    with (LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True)):
        if viz_only:
            setup(robot)
            command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        else:
            command_client = setup_and_stand(robot)

        clients = Clients(lease=lease_client, state=robot_state_client, manipulation=manipulation_api_client,
                          image=image_client, raycast=rc_client, command=command_client, robot=robot)

        rclpy.init()
        node = GenerateDataVRNode(clients, follow_arm_with_body=False, viz_only=viz_only)
        try:
            rclpy.spin(node)
        except SystemExit:
            pass

        rclpy.shutdown()
        print("Done!")


if __name__ == '__main__':
    main()
