# export BOSDYN_CLIENT_USERNAME=user && export BOSDYN_CLIENT_PASSWORD=4f9y0eftzh76

"""To get point cloud from depth image source"""

import argparse
import sys
import open3d as o3d
import cv2
import time
import numpy as np
import pdb

from conq.manipulation_lib.Manipulation import grasped_bool, open_gripper, close_gripper, move_to_blocking, move_gripper

import bosdyn.client.estop
import bosdyn.client.lease
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.api import estop_pb2, geometry_pb2, image_pb2, manipulation_api_pb2
from bosdyn.client.estop import EstopClient
from bosdyn.client.frame_helpers import VISION_FRAME_NAME, get_vision_tform_body, math_helpers
from bosdyn.client.image import ImageClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot_command import RobotCommandClient, blocking_stand
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.ray_cast import RayCastClient
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient,
                                         block_until_arm_arrives, blocking_stand)

import bosdyn.client
import bosdyn.client.util
from bosdyn.client.image import ImageClient, depth_image_to_pointcloud, _depth_image_data_to_numpy
from conq.cameras_utils import get_color_img, get_depth_img, pos_in_cam_to_pos_in_hand, image_to_opencv, RGB_SOURCES, DEPTH_SOURCES
from bosdyn.client.frame_helpers import VISION_FRAME_NAME

from conq.clients import Clients

import matplotlib.pyplot as plt

def verify_estop(robot):
    """Verify the robot is not estopped"""

    client = robot.ensure_client(EstopClient.default_service_name)
    if client.get_status().stop_level != estop_pb2.ESTOP_LEVEL_NONE:
        error_message = 'Robot is estopped. Please use an external E-Stop client, such as the' \
                        ' estop SDK example, to configure E-Stop.'
        robot.logger.error(error_message)
        raise Exception(error_message)
    
def key_callback(vis, action, mods):
    """
    Callback function for key events in Open3D visualizer.
    
    Parameters:
    - vis: The visualizer object.
    - action: The key action (pressed, released).
    - mods: Modifier keys.
    """
    # Check if 'Q' or 'Esc' is pressed
    if action.key == ord('Q') or action.key == 256:  # 256 is the key code for 'Esc'
        vis.close()  # Close the visualizer
        return False  # Return False to indicate the event has been handled

def main(argv):
    # Parse args
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument('--to-depth',
                        help='Convert to the depth frame. Default is convert to visual.',
                        action='store_true')
    parser.add_argument('--camera', help='Camera to acquire image from.', default='hand',\
                        choices=['frontleft', 'frontright', 'left', 'right', 'back','hand',
                        ])
    parser.add_argument('--auto-rotate', help='rotate right and front images to be upright',
                        action='store_true')
    options = parser.parse_args(argv)

    if options.to_depth:
        sources = [options.camera + '_depth', options.camera +'_color_image']
    else:
        sources = [options.camera + 'frontleft_depth_in_visual_frame', options.camera + '_fisheye_image']


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

    save_path = "/home/vision/Spot590/conq_python/data/"

    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        # Now, we are ready to power on the robot. This call will block until the power
        # is on. Commands would fail if this did not happen. We can also check that the robot is
        # powered at any point.
        robot.logger.info('Powering on robot... This may take a several seconds.')
        robot.power_on(timeout_sec=20)
        assert robot.is_powered_on(), 'Robot power on failed.'
        robot.logger.info('Robot powered on.')

        clients = Clients(lease=lease_client, state=robot_state_client, manipulation=manipulation_api_client,
                            image=image_client, raycast=rc_client, command=command_client, robot=robot, graphnav=None)

        
        robot.logger.info('Commanding robot to stand...')
        blocking_stand(command_client, timeout_sec=10)
        robot.logger.info('Robot standing.')

        # Deploy the arm
        robot_cmd = RobotCommandBuilder.arm_ready_command()
        cmd_id = command_client.robot_command(robot_cmd)
        block_until_arm_arrives(command_client, cmd_id)

        # Task 1: Look at scene
        gaze_pose = (0.75,0.0,0.1, 0.7071,0.,0.7071,0)
        status = move_gripper(clients, gaze_pose, blocking=False, duration = 0.5)

        status = open_gripper(clients)
        time.sleep(1)

        # Initialize Open3D visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window('Point Cloud Visualization', width=800, height=600)
        pcd = o3d.geometry.PointCloud()
        vis.add_geometry(pcd)

        # Live point cloud
        start_time = time.time()
        
        image_responses = image_client.get_image_from_sources(sources)
        image = image_responses[0]
        # Get point cloud
        pcl_xyz = depth_image_to_pointcloud(image_response=image, min_dist=0, max_dist=10)
        # Save depth image
        depth_img = _depth_image_data_to_numpy(image_response=image)
       
        
        end_time = time.time()
        print("Exeuted for: ",np.round(end_time - start_time,4))
            
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcl_xyz)
        o3d.visualization.draw_geometries([pcd])

        # Image responses are in the same order as the requests.
        # Convert to opencv images.

        object_name = input("Enter object name: ")

        np.save(f'{save_path}/NPY/{object_name}.npy',pcl_xyz)
        np.save(f'{save_path}/DEPTH/{object_name}.npy',depth_img)

        if len(image_responses) < 2:
            print('Error: failed to get images.')
            return False
        
        # Visual is a JPEG
        cv_visual = cv2.imdecode(np.frombuffer(image_responses[1].shot.image.data, dtype=np.uint8), -1)

        # Write the image out.
        filename = f'{save_path}/RGB/{object_name}.jpg'
        cv2.imwrite(filename, cv_visual)

        # Depth is a raw bytestream
        cv_depth = np.frombuffer(image_responses[0].shot.image.data, dtype=np.uint16)
        
        cv_depth = cv_depth.reshape(image_responses[0].shot.image.rows,
                                    image_responses[0].shot.image.cols)
        
        try:
            print("Shape of cv_depth: ", np.shape(cv_depth))
        except:
            print("Not an array")


        # cv2.applyColorMap() only supports 8-bit; convert from 16-bit to 8-bit and do scaling
        min_val = np.min(cv_depth)
        max_val = np.max(cv_depth)
        depth_range = max_val - min_val
        depth8 = (255.0 / depth_range * (cv_depth - min_val)).astype('uint8')
        depth8_rgb = cv2.cvtColor(depth8, cv2.COLOR_GRAY2RGB)
        


        # Write the image out.
        filename = f'{save_path}/RGB/{object_name}_depth.jpg'
        cv2.imwrite(filename, depth8_rgb)

    return True

def cv_mouse_callback(event, x, y, flags, param):
    global g_image_click, g_image_display
    clone = g_image_display.copy()
    if event == cv2.EVENT_LBUTTONUP:
        g_image_click = (x, y)
    else:
        # Draw some lines on the image.
        # print('mouse', x, y)
        color = (30, 30, 30)
        thickness = 2
        image_title = 'Click to grasp'
        height = clone.shape[0]
        width = clone.shape[1]
        cv2.line(clone, (0, y), (width, y), color, thickness)
        cv2.line(clone, (x, 0), (x, height), color, thickness)
        cv2.imshow(image_title, clone)


if __name__ == '__main__':
    if not main(sys.argv[1:]):
        sys.exit(1)
