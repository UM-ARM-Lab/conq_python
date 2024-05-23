# BOSTON DYNAMICS API
import cv2
import numpy as np
from bosdyn import geometry
from bosdyn.api import (
    arm_command_pb2,
    estop_pb2,
    geometry_pb2,
    image_pb2,
    manipulation_api_pb2,
    robot_command_pb2,
    synchronized_command_pb2,
)
from bosdyn.client import math_helpers
from bosdyn.client.estop import EstopClient
from bosdyn.client.frame_helpers import (
    GRAV_ALIGNED_BODY_FRAME_NAME,
    HAND_FRAME_NAME,
    VISION_FRAME_NAME,
    get_a_tform_b,
)
from bosdyn.client.image import (
    _depth_image_data_to_numpy,
    _depth_image_get_valid_indices,
)
from bosdyn.client.math_helpers import Quat, quat_to_eulerZYX
from bosdyn.client.robot_command import (
    RobotCommandBuilder,
    block_until_arm_arrives,
    blocking_stand,
)
from scipy.spatial.transform import Rotation as R

from conq.cameras_utils import (
    DEPTH_SOURCES,
    RGB_SOURCES,
    get_color_img,
    get_depth_img,
    image_to_opencv,
    pos_in_cam_to_pos_in_hand,
    rotate_image,
)
from conq.clients import Clients


def build_arm_target_from_vision(clients, visual_pose):
    x_hand,y_hand,z_hand,roll_hand, pitch_hand, yaw_hand = visual_pose
    roll_hand, pitch_hand, yaw_hand = 0,0,0 # FIXME: Make this generalized
    # 2. TRANSFORM from Hand pose to Grav_aligned_body pose
    # TODO: Convert euler to rot
    transforms = clients.state.get_robot_state().kinematic_state.transforms_snapshot
    # Transformation of hand w.r.t grav_aligned
    BODY_T_HAND = get_a_tform_b(transforms, GRAV_ALIGNED_BODY_FRAME_NAME, HAND_FRAME_NAME)
    
    xyz = geometry_pb2.Vec3(x = x_hand, y=y_hand,z=z_hand)
    euler = geometry.EulerZXY(roll = roll_hand, pitch = pitch_hand, yaw = yaw_hand)
    quat = euler.to_quaternion()
    HAND_T_OBJECT = geometry_pb2.SE3Pose(position = xyz, rotation = quat)
    BODY_T_OBJECT = BODY_T_HAND * math_helpers.SE3Pose.from_proto(HAND_T_OBJECT)

    #print("Object pose in Body frame",BODY_T_OBJECT)
    arm_command_pose = (BODY_T_OBJECT.position.x,BODY_T_OBJECT.position.y,BODY_T_OBJECT.position.z,BODY_T_OBJECT.rotation.w,BODY_T_OBJECT.rotation.x, BODY_T_OBJECT.rotation.y,BODY_T_OBJECT.rotation.z)
    return arm_command_pose

def make_robot_command(arm_joint_traj):
    """ Adapted from sdk example. Helper function to create a RobotCommand from an ArmJointTrajectory.
        The returned command will be a SynchronizedCommand with an ArmJointMoveCommand
        filled out to follow the passed in trajectory. """

    joint_move_command = arm_command_pb2.ArmJointMoveCommand.Request(trajectory=arm_joint_traj)
    arm_command = arm_command_pb2.ArmCommand.Request(arm_joint_move_command=joint_move_command)
    sync_arm = synchronized_command_pb2.SynchronizedCommand.Request(arm_command=arm_command)
    arm_sync_robot_cmd = robot_command_pb2.RobotCommand(synchronized_command=sync_arm)
    return RobotCommandBuilder.build_synchro_command(arm_sync_robot_cmd)

def hand_pose_cmd_to_frame(a_in_frame, x, y, z, roll, pitch, yaw):
    hand_pos_in_body = geometry_pb2.Vec3(x=x, y=y, z=z)
    euler = geometry.EulerZXY(roll=roll, pitch=pitch, yaw=yaw)
    quat_hand = euler.to_quaternion()
    hand_in_body = geometry_pb2.SE3Pose(position=hand_pos_in_body, rotation=quat_hand)
    hand_in_vision = a_in_frame * math_helpers.SE3Pose.from_proto(hand_in_body)
    return hand_in_vision

def verify_estop(robot):
    """Verify the robot is not estopped"""

    client = robot.ensure_client(EstopClient.default_service_name)
    if client.get_status().stop_level != estop_pb2.ESTOP_LEVEL_NONE:
        error_message = 'Robot is estopped. Please use an external E-Stop client, such as the' \
                        ' estop SDK example, to configure E-Stop.'
        robot.logger.error(error_message)
        raise Exception(error_message)

def unstow_arm(robot, command_client):
    stand_command = RobotCommandBuilder.synchro_stand_command() # params
    unstow = RobotCommandBuilder.arm_ready_command(build_on_command=stand_command)
    unstow_command_id = command_client.robot_command(unstow)
    robot.logger.info('Unstow command issued.')
    block_until_arm_arrives(command_client, unstow_command_id, 3.0)


def stow_arm(robot, command_client):
    stow_cmd = RobotCommandBuilder.arm_stow_command()
    stow_command_id = command_client.robot_command(stow_cmd)
    robot.logger.info('Stow command issued.')
    block_until_arm_arrives(command_client, stow_command_id, 3.0)

def stand(robot, command_client, params=None):
    robot.logger.info('Commanding robot to stand...')
    blocking_stand(command_client, timeout_sec=10, params=params)
    robot.logger.info('Robot standing.')

#### PERCEPTION UTILS ####

def depth2pcl(image_response, seg_mask = None, min_dist=0, max_dist=1000):
     """
     Convert depth image to point cloud.
     Modified from bosdyn.clients.image
     """
     if image_response.source.image_type != image_pb2.ImageSource.IMAGE_TYPE_DEPTH:
         raise ValueError('requires an image_type of IMAGE_TYPE_DEPTH')
    #  if image_response.source.pixel_format != image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
    #      raise ValueError('IMAGE_TYPE_DEPTH with an unsupported format, required PIXEL_FORMAT_DEPTH_U16')
     if not image_response.source.HasField('pinhole'):
         raise ValueError('Requires a pinhole camera model')
     
     source_rows = image_response.source.rows
     source_cols = image_response.source.cols
     fx = image_response.source.pinhole.intrinsics.focal_length.x
     fy = image_response.source.pinhole.intrinsics.focal_length.y
     cx = image_response.source.pinhole.intrinsics.principal_point.x
     cy = image_response.source.pinhole.intrinsics.principal_point.y
     depth_scale = image_response.source.depth_scale

     # Convert the proto representation into a numpy array
     depth_array = np.copy(_depth_image_data_to_numpy(image_response)) # Assignment destination is read-only

     # SEGMENT if seg_mask is not None:
     if seg_mask is not None:
         depth_array[~seg_mask] = 0  # Set non-segmented regions to zero

     # Determine which indices have valid data in the user requested range
     valid_inds = _depth_image_get_valid_indices(depth_array, np.rint(min_dist*depth_scale),
                                                 np.rint(max_dist * depth_scale))

     # Compute the valid data
     rows, cols = np.mgrid[0:source_rows, 0:source_cols]
     depth_array = depth_array[valid_inds]
     rows = rows[valid_inds]
     cols = cols[valid_inds]

     # Convert the valid distance data to (x,y,z) values expressed in the sensor frame
     z = depth_array / depth_scale
     x = np.multiply(z, (cols - cx)) / fx
     y = np.multiply(z, (rows - cy)) / fy
     return np.vstack((x,y,z)).T

def pcl_transform(pcl_xyz, image_response, source,target_frame):
    pcl_xyz1 = np.hstack((pcl_xyz,np.ones((pcl_xyz.shape[0],1))))
    # TRANSFORM FROM sensor frame to GRAV_ALIGNED frame
    BODY_T_VISION = get_a_tform_b(image_response.shot.transforms_snapshot, target_frame, "hand_color_image_sensor") # 4x4
    body_pcl_hand_sensor = np.dot(BODY_T_VISION.to_matrix(),pcl_xyz1.T).T # Nx4
    # Remove ones
    body_pcl_hand_sensor = body_pcl_hand_sensor[:,:-1] #/body_pcl_hand_sensor[:,-1][:,np.newaxis]

    return body_pcl_hand_sensor

def rotate_quaternion(pose,angle_degrees, axis=(0,1,0)):
    pose = list(pose)
    angle_radians = np.radians(angle_degrees)
    rotation = R.from_rotvec(angle_radians*np.array(axis))
    quat_matrix = R.from_quat([pose[3], pose[4], pose[5], pose[6]]).as_matrix()
    rotate_quat_matrix = rotation.apply(quat_matrix)
    rotated_quaternion = R.from_matrix(rotate_quat_matrix).as_quat()
    return tuple([pose[0],pose[1],pose[2],rotated_quaternion[0],rotated_quaternion[1],rotated_quaternion[2],rotated_quaternion[3]])


def get_segmask(shape=(480,640), center=(240,320), min_radius = 20, max_radius = 100):
    y,x = np.ogrid[:shape[0], :shape[1]]
    radius = np.random.randint(min_radius, max_radius+1, size=shape)
    dist = np.sqrt((x-center[1])**2 + (y-center[0])**2)
    segmask = dist <= radius
    return segmask

RGB_PATH = "src/conq/manipulation_lib/gpd/data/RGB/"
MASK_PATH = "src/conq/manipulation_lib/gpd/data/MASK/"

def get_segmask_manual(image_path = RGB_PATH+"live.jpg", save_path = MASK_PATH):
    # Global variables
    drawing = False  # True if mouse is pressed
    ix, iy = -1, -1  # Starting position of the drawing

    # Mouse callback function
    def draw_circle(event, x, y, flags, param):
        nonlocal ix, iy, drawing

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                cv2.circle(seg_mask, (x, y), 10, (255), -1)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            cv2.circle(seg_mask, (x, y), 10, (255), -1)

    # Load an image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image.")
        return None

    # Create a black segmentation mask
    seg_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    # Create a window and set mouse callback
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_circle)

    while True:
        # Display the image and segmentation mask
        combined_image = cv2.addWeighted(image, 0.6, cv2.cvtColor(seg_mask, cv2.COLOR_GRAY2BGR), 0.4, 0)
        cv2.imshow('image', combined_image)

        # Wait for key press
        key = cv2.waitKey(1) & 0xFF

        # If 'r' is pressed, reset the segmentation mask
        if key == ord('r'):
            seg_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        # If 's' is pressed, save the segmentation mask and exit
        elif key == ord('s'):
            cv2.imwrite(save_path+"live_mask.jpg", seg_mask)
            break

        # If 'q' is pressed, exit without saving the segmentation mask
        elif key == ord('q'):
            return None

    # Cleanup
    cv2.destroyAllWindows()

    # Convert segmentation mask to True/False array
    segmask = seg_mask.astype(bool)

    return segmask

## Image and AprilTag testing codes:
# TODO: Move to utils
def get_Image(image_client, sources):
    rgb_np, rgb_response = get_color_img(image_client, sources)
    rgb_np = np.array(rgb_np, dtype=np.uint8)

    img_bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR) 
    return img_bgr, rgb_response
    
def get_object_pose(img_bgr, rgb_response, camera_params):
            
    result, overlay = apriltag_image(img_bgr, camera_params, tag_size=0.0635) # Replace with object_pose estimator

    ### ______________________###
    object_pose = result[1]
                
    detection = result[0]
    
    #print("Object pose: \n", object_pose)
    #print("Object in camera frame: \n",object_pose[:-1,-1])

    # TODO / FIXME: Use transformation instead of shortcut for pose
    # Transformation from Camera Pose to Hand Pose
    object_in_hand = np.copy(object_pose)
    hand_T_camera = np.array([[0,0,1,0],
                              [-1,0,0,0],
                              [0,-1,0,0],
                              [0,0,0,1]])
    
    object_in_hand = np.matmul(hand_T_camera,object_pose)
    print("Object in hand: \n", object_in_hand)

    quat = Quat.from_matrix(object_in_hand[:-1,:-1])
    print("Rotation in quaternion: ", quat)

    yaw_hand, pitch_hand,roll_hand = quat_to_eulerZYX(quat)

    # object_in_hand[0,-1]  = object_pose[2,-1] # X_hand = Z_camera
    # object_in_hand[1,-1]  = -object_pose[0,-1] # Y_hand = -X_camera
    # object_in_hand[2,-1]  = -object_pose[1,-1] # Z_hand = -Y_camera
    
    #print("Object in hand frame: \n",object_in_hand[:-1,-1])
    x_hand,y_hand,z_hand = object_in_hand[0,-1],object_in_hand[1,-1],object_in_hand[2,-1]
    #roll_hand,pitch_hand,yaw_hand = 0,0,0
    visual_pose = (x_hand,y_hand,z_hand,roll_hand,pitch_hand,yaw_hand)
    #visual_pose = (x_hand,y_hand,z_hand,quat.w,quat.x,quat.y,quat.z)

    ## Overlays
    object_center = (int(detection.center[0]),int(detection.center[1]))   # Assuming this gives the center as [x, y]

    # Draw circular marker at object center
    cv2.circle(overlay, object_center, 5, (0, 255, 0), -1)

    # Image center
    image_center = [int(overlay.shape[1] // 2), int(overlay.shape[0] // 2)]
    cv2.circle(overlay, tuple(image_center), 5, (255, 0, 0), -1)

    # Draw arrow from image center to object center
    cv2.arrowedLine(overlay, tuple(image_center), tuple(object_center), (0, 0, 255), 2)

    return result, rgb_response, overlay, visual_pose, object_center

def apriltag_image(img, camera_params, tag_size):

    '''
    Detect AprilTags from static images.

    Args:   input_images: cv image
            camera_params [list[float]]: [fx,fy,cx,cy]
    '''


    '''
    Set up a reasonable search path for the apriltag DLL.
    Either install the DLL in the appropriate system-wide
    location, or specify your own search paths as needed.
    '''

    detector = apriltag.Detector(options=None, searchpath=apriltag._get_dll_path())

    result, overlay = apriltag.detect_tags(img,
                                            detector,
                                            camera_params,
                                            tag_size,
                                            vizualization=3,
                                            verbose=None,
                                            annotation=True
                                            )
    return result, overlay
