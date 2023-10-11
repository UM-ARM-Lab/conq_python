import time

import bosdyn
import numpy as np
import rerun as rr
from bosdyn.api import arm_command_pb2, synchronized_command_pb2, basic_command_pb2, mobility_command_pb2
from bosdyn.api import robot_command_pb2
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import get_a_tform_b, VISION_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME
from bosdyn.client.image import ImageClient
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.ray_cast import RayCastClient
from bosdyn.client.robot_command import RobotCommandBuilder
from bosdyn.client.robot_state import RobotStateClient

from conq.clients import Clients
from conq.rerun_utils import viz_common_frames
from conq.utils import setup_and_stand


def main():
    np.set_printoptions(precision=3, suppress=True)

    rr.init("arm_traj_test")
    rr.connect()

    sdk = bosdyn.client.create_standard_sdk('arm_traj_test')
    robot = sdk.create_robot('192.168.80.3')
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()

    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    manipulation_api_client = robot.ensure_client(ManipulationApiClient.default_service_name)
    image_client = robot.ensure_client(ImageClient.default_service_name)
    rc_client = robot.ensure_client(RayCastClient.default_service_name)

    lease_client.take()

    with (LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True)):
        command_client = setup_and_stand(robot)
        robot.start_time_sync(1)
        time.sleep(1)

        clients = Clients(lease=lease_client, state=robot_state_client, manipulation=manipulation_api_client,
                          image=image_client, raycast=rc_client, command=command_client, robot=robot,
                          recorder=None)

        # poses are relative to the base at the start of the program
        T = 50
        poses = np.array([
                             [0, 0, 0.5, 0, 0, 0],
                         ] * T)
        # ramp the x coordinate from 0.5 to 1.5 then back down
        t = np.linspace(0, 1, T)
        poses[:, 0] = np.sin(t * np.pi) + 0.75

        # ready
        cmd = RobotCommandBuilder.arm_ready_command()
        clients.command.robot_command(cmd)

        transforms0 = clients.state.get_robot_state().kinematic_state.transforms_snapshot
        body_in_vision0 = get_a_tform_b(transforms0, VISION_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME)

        dt = 0.1
        for pose in poses:
            x, y, z, roll, pitch, yaw = pose

            cartesian_velocity = arm_command_pb2.ArmVelocityCommand.CartesianVelocity()
            cartesian_velocity.frame_name = VISION_FRAME_NAME
            cartesian_velocity.velocity_in_frame_name.x = -0.1
            cartesian_velocity.velocity_in_frame_name.y = 0.05
            cartesian_velocity.velocity_in_frame_name.z = -0.05
            end_time = bosdyn.util.seconds_to_timestamp(robot.time_sec() + 2 * dt)
            arm_vel_req = arm_command_pb2.ArmVelocityCommand.Request(cartesian_velocity=cartesian_velocity,
                                                                     end_time=end_time)
            arm_cmd = arm_command_pb2.ArmCommand.Request(arm_velocity_command=arm_vel_req)

            mobility_command = mobility_command_pb2.MobilityCommand.Request(
                follow_arm_request=basic_command_pb2.FollowArmCommand.Request())

            synch_cmd = synchronized_command_pb2.SynchronizedCommand.Request(arm_command=arm_cmd,
                                                                             mobility_command=mobility_command)
            cmd = robot_command_pb2.RobotCommand(synchronized_command=synch_cmd)
            clients.command.robot_command(cmd)

            time.sleep(dt)

        state = clients.state.get_robot_state()
        viz_common_frames(state.kinematic_state.transforms_snapshot)


if __name__ == '__main__':
    main()
