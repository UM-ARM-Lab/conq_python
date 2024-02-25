# export PYTHONPATH="${PYTHONPATH}:/home/vision/Desktop/conq_python/src"
# export BOSDYN_CLIENT_USERNAME=user && export BOSDYN_CLIENT_PASSWORD=4f9y0eftzh76

"""
Performs eye in hand (eih) pose based visual servoing (pbvs). 
Written by Jemuel Stanley Premkumar (jemprem@umich.edu)
"""

# PYTHON 
import numpy as np
import sys
import cv2
import time

# BOSDYN: Clients
import bosdyn.client.estop
import bosdyn.client.lease
from bosdyn.client.image import ImageClient
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient,
                                         block_until_arm_arrives, blocking_stand)
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.ray_cast import RayCastClient
from bosdyn.client.lease import LeaseClient

#BOSDYN: Helpers
from bosdyn.client.frame_helpers import VISION_FRAME_NAME, HAND_FRAME_NAME,GRAV_ALIGNED_BODY_FRAME_NAME

# CONQ: Clients
from conq.clients import Clients

# CONQ: Manipulation modules
from conq.manipulation_lib.Manipulation import move_to_unblocking, open_gripper, move_to_blocking, close_gripper
from conq.manipulation_lib.VisualServo import VisualServoingController
from conq.manipulation_lib.Perception3D import VisualPoseAcquirer

# CONQ: Utils
from conq.manipulation_lib.utils import verify_estop, build_arm_target_from_vision

# Usage:
if __name__ == "__main__":
    

    sources = ['hand' + '_depth', 'hand' +'_color_image']

    # Create robot object with an image client.
    sdk = bosdyn.client.create_standard_sdk('acquire_point_cloud')
    robot = sdk.create_robot('192.168.80.3')
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()

    assert robot.has_arm(), 'Robot requires an arm to run this example.'

    # Verify the robot is not estopped and that an external application has registered and holds
    # an estop endpoint.
    verify_estop(robot)

    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    manipulation_api_client = robot.ensure_client(ManipulationApiClient.default_service_name)
    image_client = robot.ensure_client(ImageClient.default_service_name)
    rc_client = robot.ensure_client(RayCastClient.default_service_name)
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)

    lease_client.take()

    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        # Now, we are ready to power on the robot. This call will block until the power
        # is on. Commands would fail if this did not happen. We can also check that the robot is
        # powered at any point.
        robot.logger.info('Powering on robot... This may take a several seconds.')
        robot.power_on(timeout_sec=20)
        assert robot.is_powered_on(), 'Robot power on failed.'
        robot.logger.info('Robot powered on.')

        clients = Clients(lease=lease_client, state=robot_state_client, manipulation=manipulation_api_client,
                            image=image_client, raycast=rc_client, command=command_client, robot=robot,
                            recorder=None)

        
        robot.logger.info('Commanding robot to stand...')
        blocking_stand(command_client, timeout_sec=10)
        robot.logger.info('Robot standing.')

        # Deploy the arm
        robot_cmd = RobotCommandBuilder.arm_ready_command()
        cmd_id = command_client.robot_command(robot_cmd)
        block_until_arm_arrives(command_client, cmd_id)
    
        # Task 1: Look at scene
        gaze_pose = (0.75,0.0,0.4, 1.,0.,0.,0)
        status = move_to_unblocking(clients, gaze_pose, frame_name = GRAV_ALIGNED_BODY_FRAME_NAME, duration = 0.1)

        status = open_gripper(clients)
        time.sleep(1)

        # VISUAL POSE ACQUIRER:
         # Initialize and start the visual pose acquirer thread
        camera_params = [552.0291012161067, 552.0291012161067, 320.0, 240.0]
        pose_acquirer = VisualPoseAcquirer(image_client, sources, camera_params)
        pose_acquirer.start()


        # Initialize and start the PID controller thread
        # kp = 1
        # kd = 0.01

        # target_position = current_object_pose
        # control_rate = 30  # Hz
        # TODO: Needs robot's current pose
        # gains = (kp,kd)
        # controller = VisualServoingController(gains, control_rate, pose_acquirer,clients)
        # controller.start()
        #pdb.set_trace()
        
        while True:
            try:
                current_object_pose,_ = pose_acquirer.get_latest_pose()
                if np.all(np.array(current_object_pose) == 0):
                    print("Object not in frame: ", current_object_pose)
                else:
                    arm_command_pose = build_arm_target_from_vision(clients,current_object_pose)
                    status = move_to_unblocking(clients, arm_command_pose, frame_name = GRAV_ALIGNED_BODY_FRAME_NAME, duration = 0.1)

                # Update if controller
    
            except KeyboardInterrupt:
                break
            
            except Exception as e:
                print(e)
                break
                
        print("Stopping threads.")
        pose_acquirer.stop()
        #controller.stop()
        pose_acquirer.join()
        #controller.join()

        # Stow the arm
        # Build the stow command using RobotCommandBuilder
        stow = RobotCommandBuilder.arm_stow_command()

        # Issue the command via the RobotCommandClient
        stow_command_id = command_client.robot_command(stow)

        robot.logger.info('Stow command issued.')
        block_until_arm_arrives(command_client, stow_command_id, 3.0)



