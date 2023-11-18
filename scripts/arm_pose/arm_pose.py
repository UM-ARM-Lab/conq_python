""" 
Script that allows you to control the arm through the python api (using OpenCV sliders)
and prints the orientations to the screen for ease of use

Based on arm_simple.py example
"""

from __future__ import print_function

import argparse
import sys
import time

import bosdyn.api.gripper_command_pb2
import bosdyn.client
import bosdyn.client.lease
import bosdyn.client.util
from bosdyn.api import arm_command_pb2, geometry_pb2
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import GRAV_ALIGNED_BODY_FRAME_NAME, ODOM_FRAME_NAME, get_a_tform_b
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient,
                                         block_until_arm_arrives, blocking_stand)
from bosdyn.client.robot_state import RobotStateClient

import cv2

# Define a function to do nothing (used for the trackbars)
def nothing(x):
    pass


def print_arm_pose(config):
    """A simple example of using the Boston Dynamics API to command Spot's arm."""

    # See hello_spot.py for an explanation of these lines.
    bosdyn.client.util.setup_logging(config.verbose)

    sdk = bosdyn.client.create_standard_sdk('HelloSpotClient')
    robot = sdk.create_robot(config.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()

    assert robot.has_arm(), "Robot requires an arm to run this example."

    # Verify the robot is not estopped and that an external application has registered and holds
    # an estop endpoint.
    assert not robot.is_estopped(), "Robot is estopped. Please use an external E-Stop client, " \
                                    "such as the estop SDK example, to configure E-Stop."

    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)

    lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        # Now, we are ready to power on the robot. This call will block until the power
        # is on. Commands would fail if this did not happen. We can also check that the robot is
        # powered at any point.
        robot.logger.info("Powering on robot... This may take a several seconds.")
        robot.power_on(timeout_sec=20)
        assert robot.is_powered_on(), "Robot power on failed."
        robot.logger.info("Robot powered on.")

        # Tell the robot to stand up. The command service is used to issue commands to a robot.
        # The set of valid commands for a robot depends on hardware configuration. See
        # RobotCommandBuilder for more detailed examples on command building. The robot
        # command service requires timesync between the robot and the client.
        robot.logger.info("Commanding robot to stand...")
        command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        blocking_stand(command_client, timeout_sec=10)
        robot.logger.info("Robot standing.")



        ###############
        """ 
        Move the arm using OpenCV sliders and print the pose to the screen
        """
        ################

        # Limits of the arm (meters) #TODO: get the actual limits
        xmin = 2.0
        xmax = -2.0
        ymin = -1.0
        ymax = 1.0
        zmin = -1.0
        zmax = 1.0


        # Move the arm to a spot in front of the robot, and open the gripper.

        # Make the arm pose RobotCommand
        x = 0.75
        y = 0
        z = 0.25
        
        qw = 1
        qx = 0
        qy = 0
        qz = 0

        gripper_fraction = 0.0

        cv2.namedWindow('Arm Pose', cv2.WINDOW_NORMAL)
        cv2.createTrackbar("X", "Arm Pose", x, xmax, nothing)
        cv2.createTrackbar("Y", "Arm Pose", y, ymax, nothing)
        cv2.createTrackbar("Z", "Arm Pose", z, zmax, nothing)
        cv2.createTrackbar("qw", "Arm Pose", qw, 1.0, nothing)
        cv2.createTrackbar("qx", "Arm Pose", qx, 1.0, nothing)
        cv2.createTrackbar("qy", "Arm Pose", qy, 1.0, nothing)
        cv2.createTrackbar("qz", "Arm Pose", qz, 1.0, nothing)
        cv2.createTrackbar("gripper", "Arm Pose", gripper_fraction, 1.0, nothing)

        while True:
            # Get the current positions of the trackbars
            x = cv2.getTrackbarPos("X", "Arm Pose")
            y = cv2.getTrackbarPos("Y", "Arm Pose")
            z = cv2.getTrackbarPos("Z", "Arm Pose")
            qw = cv2.getTrackbarPos("qw", "Arm Pose")
            qx = cv2.getTrackbarPos("qx", "Arm Pose")
            qy = cv2.getTrackbarPos("qy", "Arm Pose")
            qz = cv2.getTrackbarPos("qz", "Arm Pose")
            gripper_fraction = cv2.getTrackbarPos("gripper", "Arm Pose")
            # check the limits
            x = max(min(x, xmax), xmin)
            y = max(min(y, ymax), ymin)
            z = max(min(z, zmax), zmin)
            qw = max(min(qw, 1.0), -1.0)
            qx = max(min(qx, 1.0), -1.0)
            qy = max(min(qy, 1.0), -1.0)
            qz = max(min(qz, 1.0), -1.0)
            gripper_fraction = max(min(gripper_fraction, 1.0), 0.0)

            # Convert to a boston dynamics pose
            hand_ewrt_flat_body = geometry_pb2.Vec3(x=x, y=y, z=z)
            flat_body_Q_hand = geometry_pb2.Quaternion(w=qw, x=qx, y=qy, z=qz)

            flat_body_T_hand = geometry_pb2.SE3Pose(position=hand_ewrt_flat_body,
                                                    rotation=flat_body_Q_hand)

            robot_state = robot_state_client.get_robot_state()
            odom_T_flat_body = get_a_tform_b(robot_state.kinematic_state.transforms_snapshot,
                                            ODOM_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME)

            odom_T_hand = odom_T_flat_body * math_helpers.SE3Pose.from_obj(flat_body_T_hand)

            seconds = 2

            arm_command = RobotCommandBuilder.arm_pose_command(
                odom_T_hand.x, odom_T_hand.y, odom_T_hand.z, odom_T_hand.rot.w, odom_T_hand.rot.x,
                odom_T_hand.rot.y, odom_T_hand.rot.z, ODOM_FRAME_NAME, seconds)

            # Make the open gripper RobotCommand
            gripper_command = RobotCommandBuilder.claw_gripper_open_fraction_command(1.0)

            # Combine the arm and gripper commands into one RobotCommand
            command = RobotCommandBuilder.build_synchro_command(gripper_command, arm_command)

            # Send the request
            cmd_id = command_client.robot_command(command)
            robot.logger.info('Moving arm to position 1.')

            # Wait until the arm arrives at the goal.
            block_until_arm_arrives_with_prints(robot, command_client, cmd_id)

            # Print the current arm pose
            robot_state = robot_state_client.get_robot_state()
            odom_T_flat_body = get_a_tform_b(robot_state.kinematic_state.transforms_snapshot,
                                            ODOM_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME)
            print(f"Current arm pose (x,y,x,qx,qy,qz,qw,g): ({odom_T_flat_body.x}, {odom_T_flat_body.y}, {odom_T_flat_body.z}), 
                    ({odom_T_flat_body.rot.x}, {odom_T_flat_body.rot.y}, {odom_T_flat_body.rot.z}, {odom_T_flat_body.rot.w}), {gripper_fraction}")
            # time.sleep(0.1)


        # if the user presses 'q', exit the program
        if cv2.waitKey(1) & 0xFF == ord('q'):

            # Power the robot off. By specifying "cut_immediately=False", a safe power off command
            # is issued to the robot. This will attempt to sit the robot before powering off.
            robot.power_off(cut_immediately=False, timeout_sec=20)
            assert not robot.is_powered_on(), "Robot power off failed."
            robot.logger.info("Robot safely powered off.")
            return


def block_until_arm_arrives_with_prints(robot, command_client, cmd_id):
    """Block until the arm arrives at the goal and print the distance remaining.
        Note: a version of this function is available as a helper in robot_command
        without the prints.
    """
    while True:
        feedback_resp = command_client.robot_command_feedback(cmd_id)
        robot.logger.info(
            'Distance to go: ' +
            '{:.2f} meters'.format(feedback_resp.feedback.synchronized_feedback.arm_command_feedback
                                   .arm_cartesian_feedback.measured_pos_distance_to_goal) +
            ', {:.2f} radians'.format(
                feedback_resp.feedback.synchronized_feedback.arm_command_feedback.
                arm_cartesian_feedback.measured_rot_distance_to_goal))

        if feedback_resp.feedback.synchronized_feedback.arm_command_feedback.arm_cartesian_feedback.status == arm_command_pb2.ArmCartesianCommand.Feedback.STATUS_TRAJECTORY_COMPLETE:
            robot.logger.info('Move complete.')
            break
        time.sleep(0.1)


def main(argv):
    """Command line interface."""
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    options = parser.parse_args(argv)
    try:
        print_arm_pose(options)
        return True
    except Exception as exc:  # pylint: disable=broad-except
        logger = bosdyn.client.util.get_logger()
        logger.exception("Threw an exception")
        return False


if __name__ == '__main__':
    if not main(sys.argv[1:]):
        sys.exit(1)
