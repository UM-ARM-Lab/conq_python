#!/usr/bin/env python3

from rclpy.node import Node
import argparse
import time
from pathlib import Path

import bosdyn.client
import bosdyn.client.util
import numpy as np
import rclpy
from bosdyn.client.frame_helpers import get_a_tform_b, VISION_FRAME_NAME, HAND_FRAME_NAME
from bosdyn.client.image import ImageClient
from bosdyn.client.lease import LeaseKeepAlive, LeaseClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.math_helpers import SE3Pose, Quat
from bosdyn.client.ray_cast import RayCastClient
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient
from bosdyn.client.robot_state import RobotStateClient
from geometry_msgs.msg import Pose
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from turtlesim.msg import Pose

from conq.clients import Clients
from conq.data_recorder import ConqDataRecorder
from conq.hand_motion import hand_pose_cmd
from conq.manipulation import blocking_arm_command, add_follow_with_body
from conq.manipulation import open_gripper
from conq.utils import setup_and_stand, setup
from vr_ros2_bridge_msgs.msg import ControllersInfo, ControllerInfo

VR_FRAME_NAME = 'vr'


def controller_info_to_se3_pose(controller_info):
    pose_msg: Pose = controller_info.controller_pose
    return SE3Pose(x=pose_msg.position.x,
                   y=pose_msg.position.y,
                   z=pose_msg.position.z,
                   rot=Quat(w=pose_msg.orientation.w,
                            x=pose_msg.orientation.x,
                            y=pose_msg.orientation.y,
                            z=pose_msg.orientation.z))


class GenerateDataVRNode(Node):

    def __init__(self, recorder: ConqDataRecorder, conq_clients: Clients, follow_arm_with_body=True, viz_only=False):
        super().__init__("generate_data_from_vr")
        self.follow_arm_with_body = follow_arm_with_body
        self.recorder = recorder
        self.conq_clients = conq_clients  # Node already has a clients attribute, hence the name change
        self.viz_only = viz_only
        self.hand_in_vision0 = SE3Pose.from_identity()
        self.controller_in_vr0 = SE3Pose.from_identity()

        self.tf_broadcaster = TransformBroadcaster(self)
        self.vr_sub = self.create_subscription(ControllersInfo, "vr_controller_info", self.on_controllers_info, 10)

        self.has_started = False
        self.is_recording = False
        self.is_done = False

    def send_cmd(self, target_hand_in_vision: SE3Pose, open_fraction: float):
        hand_pose_msg = target_hand_in_vision.to_proto()
        arm_cmd = RobotCommandBuilder.arm_pose_command_from_pose(hand_pose_msg, VISION_FRAME_NAME, 0.1)
        if self.follow_arm_with_body:
            arm_body_cmd = add_follow_with_body(arm_cmd)
        else:
            arm_body_cmd = arm_cmd
        # arm_body_cmd.synchronized_command.arm_command.arm_cartesian_command.max_linear_velocity.value = 0.3
        # arm_body_cmd.synchronized_command.arm_command.arm_cartesian_command.max_angular_velocity.value = 000

        gripper_cmd = RobotCommandBuilder.claw_gripper_open_fraction_command(open_fraction)
        arm_body_gripper_cmd = RobotCommandBuilder.build_synchro_command(arm_body_cmd, gripper_cmd)

        self.conq_clients.command.robot_command(arm_body_gripper_cmd)

    def on_done(self):
        self.recorder.stop()

        open_gripper(self.conq_clients)
        blocking_arm_command(self.conq_clients, RobotCommandBuilder.arm_stow_command())

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

        if self.has_started and controller_info.trackpad_button:
            self.is_done = True
            self.on_done()

        if self.is_recording:
            # for debugging purposes, we publish a fixed transform from "VR" frame to "VISION" frame,
            # even though we don't ever actually use this transform for controlling the robot.
            vr_to_vision = TransformStamped()
            vr_to_vision.header.stamp = self.get_clock().now().to_msg()
            vr_to_vision.header.frame_id = VR_FRAME_NAME
            vr_to_vision.child_frame_id = VISION_FRAME_NAME
            vr_to_vision.transform.translation.x = 1.5
            vr_to_vision.transform.rotation.w = 1.
            self.tf_broadcaster.sendTransform(vr_to_vision)

            # translate delta pose in VR frame to delta pose in VISION frame and send a command to the robot!
            current_controller_in_vr = controller_info_to_se3_pose(controller_info)
            self.pub_se3_pose_to_tf(current_controller_in_vr, 'current VR', VR_FRAME_NAME)
            delta_in_vr = self.controller_in_vr0.inverse() * current_controller_in_vr
            delta_in_vision = delta_in_vr  # NOTE: may need a rotation here
            target_hand_in_vision = self.hand_in_vision0 * delta_in_vision

            snapshot = self.conq_clients.state.get_robot_state().kinematic_state.transforms_snapshot
            hand_in_vision = get_a_tform_b(snapshot, VISION_FRAME_NAME, HAND_FRAME_NAME)

            open_fraction = 1 - controller_info.trigger_axis

            self.pub_se3_pose_to_tf(self.controller_in_vr0, 'controller_in_vr0', VR_FRAME_NAME)
            self.pub_se3_pose_to_tf(self.hand_in_vision0, 'hand_in_vision0', VISION_FRAME_NAME)
            self.pub_se3_pose_to_tf(hand_in_vision, 'hand_in_vision', VISION_FRAME_NAME)
            self.pub_se3_pose_to_tf(target_hand_in_vision, 'target_in_vision', VISION_FRAME_NAME)

            if not self.viz_only:
                self.send_cmd(target_hand_in_vision, open_fraction)

    def on_start_recording(self, controller_info: ControllerInfo):
        # Store the initial pose of the hand in vision frame, as well as the controller pose which is in VR frame
        print(f"Starting recording episode {self.recorder.episode_idx}")

        self.controller_in_vr0 = controller_info_to_se3_pose(controller_info)
        snapshot = self.conq_clients.state.get_robot_state().kinematic_state.transforms_snapshot
        self.hand_in_vision0 = get_a_tform_b(snapshot, VISION_FRAME_NAME, HAND_FRAME_NAME)

        mode = "unsorted"
        self.recorder.start_episode(mode)

    def on_stop_recording(self):
        print(f"Stopping recording episode {self.recorder.episode_idx}")
        self.recorder.next_episode()

    def pub_se3_pose_to_tf(self, pose: SE3Pose, child_frame_name: str, parent_frame_name:str):
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


def main():
    np.seterr(all='raise')
    np.set_printoptions(precision=3, suppress=True)
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)

    # Creates client, robot, and authenticates, and time syncs
    viz_only = False
    sdk = bosdyn.client.create_standard_sdk('generate_dataset')
    robot = sdk.create_robot('192.168.80.3')
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()

    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    manipulation_api_client = robot.ensure_client(ManipulationApiClient.default_service_name)
    image_client = robot.ensure_client(ImageClient.default_service_name)
    rc_client = robot.ensure_client(RayCastClient.default_service_name)

    lease_client.take()

    now = int(time.time())
    root = Path(f"data/conq_vr_data_{now}")
    recorder = ConqDataRecorder(root, robot_state_client, image_client, sources=[
        'hand_color_image',
        'frontleft_fisheye_image',
        'frontright_fisheye_image',
    ])

    with (LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True)):
        if viz_only:
            setup(robot)
            command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        else:
            command_client = setup_and_stand(robot)

        clients = Clients(lease=lease_client, state=robot_state_client, manipulation=manipulation_api_client,
                          image=image_client, raycast=rc_client, command=command_client, robot=robot,
                          recorder=recorder)

        if not viz_only:
            look_cmd = hand_pose_cmd(clients, 0.6, 0, 0.6, 0, np.deg2rad(0), 0, duration=0.5)
            blocking_arm_command(clients, look_cmd)
            open_gripper(clients)

        rclpy.init(args=None)
        node = GenerateDataVRNode(recorder, clients, follow_arm_with_body=True, viz_only=viz_only)
        rclpy.spin(node)


if __name__ == '__main__':
    main()
