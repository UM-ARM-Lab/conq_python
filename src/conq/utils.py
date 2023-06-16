import time

import numpy as np
from bosdyn.api import manipulation_api_pb2, arm_command_pb2, synchronized_command_pb2, robot_command_pb2
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient, blocking_stand, block_until_arm_arrives


def blocking_arm_command(command_client, cmd):
    block_until_arm_arrives(command_client, command_client.robot_command(cmd))
    # FIXME: why is this needed???
    command_client.robot_command(RobotCommandBuilder.stop_command())


def block_for_manipulation_api_command(robot, manipulation_api_client, cmd_response):
    while True:
        time.sleep(0.25)
        feedback_request = manipulation_api_pb2.ManipulationApiFeedbackRequest(
            manipulation_cmd_id=cmd_response.manipulation_cmd_id)

        # Send the request
        response = manipulation_api_client.manipulation_api_feedback_command(
            manipulation_api_feedback_request=feedback_request)

        state_name = manipulation_api_pb2.ManipulationFeedbackState.Name(response.current_state)
        robot.logger.info(f'Current state: {state_name}')

        if response.current_state == manipulation_api_pb2.MANIP_STATE_DONE:
            break

    robot.logger.info('Finished.')


def make_robot_command(arm_joint_traj):
    """ Helper function to create a RobotCommand from an ArmJointTrajectory.
        The returned command will be a SynchronizedCommand with an ArmJointMoveCommand
        filled out to follow the passed in trajectory. """

    joint_move_command = arm_command_pb2.ArmJointMoveCommand.Request(trajectory=arm_joint_traj)
    arm_command = arm_command_pb2.ArmCommand.Request(arm_joint_move_command=joint_move_command)
    sync_arm = synchronized_command_pb2.SynchronizedCommand.Request(arm_command=arm_command)
    arm_sync_robot_cmd = robot_command_pb2.RobotCommand(synchronized_command=sync_arm)
    return RobotCommandBuilder.build_synchro_command(arm_sync_robot_cmd)


def setup_and_stand(robot):
    robot.logger.info("Powering on robot... This may take a several seconds.")
    robot.power_on(timeout_sec=20)
    assert robot.is_powered_on(), "Robot power on failed."
    robot.logger.info("Robot powered on.")
    robot.logger.info("Commanding robot to stand...")
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)
    blocking_stand(command_client, timeout_sec=10)
    robot.logger.info("Robot standing.")
    return command_client


def rotate_image_coordinates(pts, width, height, angle):
    """
    Rotate image coordinates by rot degrees around the center of the image.

    Args:
        pts: Nx2 array of image coordinates
        width: width of image
        height: height of image
        angle: rotation in degrees
    """
    center = np.array([width / 2, height / 2])
    new_pts = center + (pts - center) @ rot_2d(np.deg2rad(angle)).T
    return new_pts


def rot_2d(angle):
    return np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ])


def open_gripper(command_client):
    command_client.robot_command(RobotCommandBuilder.claw_gripper_open_command())
    time.sleep(1)  # FIXME: how to block on a gripper command?
