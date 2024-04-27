# PYTHON
import math
import time

# BOSDYN: API
import bosdyn
import numpy as np
import rerun as rr
from bosdyn.api import (
    arm_command_pb2,
    estop_pb2,
    geometry_pb2,
    robot_command_pb2,
    synchronized_command_pb2,
)
from bosdyn.api.spot.inverse_kinematics_pb2 import (
    InverseKinematicsRequest,
    InverseKinematicsResponse,
)

# BOSDYN: Clients
from bosdyn.client.frame_helpers import (
    BODY_FRAME_NAME,
    GRAV_ALIGNED_BODY_FRAME_NAME,
    GROUND_PLANE_FRAME_NAME,
    HAND_FRAME_NAME,
    VISION_FRAME_NAME,
    WR1_FRAME_NAME,
    get_a_tform_b,
)
from bosdyn.client.math_helpers import Quat, SE3Pose
from bosdyn.client.ray_cast import RayCastClient
from bosdyn.client.robot_command import (
    RobotCommandBuilder,
    RobotCommandClient,
    block_until_arm_arrives,
    blocking_stand,
)

# BODSYN: Utils
from bosdyn.util import duration_to_seconds, seconds_to_duration
from google.protobuf import wrappers_pb2

# CONQ: Clients
from conq.clients import Clients
from conq.hand_motion import (
    hand_pose_cmd,
    hand_pose_cmd_in_frame,
    hand_pose_cmd_to_vision,
)

# CONQ: Wrappers
from conq.manipulation import (  # , open_gripper
    add_follow_with_body,
    blocking_arm_command,
    get_is_grasping,
)
from conq.manipulation_lib.Grasp import get_best_grasp_pose, get_grasp_candidates


# CONQ: Utils
# CONQ: Manipulation 
from conq.manipulation_lib.utils import (
    get_segmask,
    get_segmask_manual,
    hand_pose_cmd_to_frame,
    make_robot_command,
    rotate_quaternion,
    verify_estop,
)
from conq.utils import setup_and_stand


def grasped_bool(clients: Clients):
    #FIXME: Needs force based instead of position based
    #FIXME: Replace with rob_state.client.get_rob_state().manipulation_state.is_gripper_holding_item
    is_grasping = get_is_grasping(clients)
    return is_grasping
    
def move_gripper(clients: Clients, pose, blocking = True, frame_name = BODY_FRAME_NAME, duration = 1, follow=False):
    """
    Arm Move command to a pose relative to the frame name

    Args:
        clients: Clients
        pose       :
                    x: position in meters in front of the body center
                    y: position in meters to the left of the body center
                    z: position in meters above the body center
                    qw, qx,qy,qz : Orientation in quaternion
        blocking   :
                    True: Synchronous(blocked) 
                    False: Asynchronous(unblocked) 
                    default -> True
        frame_name : Command pose frame
                    default -> GRAV_ALIGNED_BODY_FRAME_NAME

        duration   : duration in seconds
        follow     : Calculates reachability based on Inverse kinematics
                    default -> False

    FUTURE:
    - Body assist 

    """
    
    try:
        print("Arm command received")
        x,y,z,qw,qx,qy,qz = pose
        arm_command = RobotCommandBuilder.arm_pose_command(
            x, y, z, qw, qx,qy, qz, frame_name, duration,False)
        
        if blocking:
            cmd_id = clients.command.robot_command(command=arm_command,end_time_secs=duration, timesync_endpoint=duration+0.25,lease=None)
            status = block_until_arm_arrives(clients.command,cmd_id)
        else:
            cmd_id = clients.command.robot_command_async(command = arm_command, end_time_secs=duration, timesync_endpoint=duration+0.25, lease=None)
            # TODO: Get feedback
            status = True
        
        # TODO: Check for reachability
        # if follow: 
        #     #TODO: determmined by reachability
        #     arm_cmd = add_follow_with_body(arm_cmd)
        return status
    except Exception as e:
        print(e)
        return False
    
def open_gripper(clients: Clients):
    try:
        clients.command.robot_command(RobotCommandBuilder.claw_gripper_open_command())
        time.sleep(1)  # FIXME: how to block on a gripper command?
        return True
    except:
        return False
    
def close_gripper(clients: Clients):
    try:
        clients.command.robot_command(RobotCommandBuilder.claw_gripper_close_command())
        time.sleep(1)  # FIXME: how to block on a gripper command?
        return True
    except:
        return False

def perform_grasp(clients: Clients, blocking=True):
    """
    Perform grasp action
    """
    try:
        # Call Grasp detection Module
        grasp_pose = get_best_grasp_pose()
        modified_pose = list(grasp_pose)
        # modified_pose[0]-=0.30
        new_grasp_pose = tuple(modified_pose)

        rotated_pose = rotate_quaternion(new_grasp_pose,-90,axis=(0,1,0))

        # Execute grasp
        status = move_gripper(clients, rotated_pose, blocking = True, duration = 1)
        status = close_gripper(clients)
        return status
    except Exception as e:
        print(e)
        return False

def move_joint_trajectory(clients: Clients,joint_targets, max_vel=2.5, max_acc=15, times=[2.0,4.0]):
    """
    Function to execute joint trajectory
    Args:
    - clients: Clients
    - joint_targets: List of [tuple(joint)]
    - max_vel: Maximum velocity for all the joints
    - max_acc: Maximum acceleration constraint for all the joints
    - time (list): Reach trajectory at time step
    """
    print("Entered joint trajectory execution")
    max_vel = wrappers_pb2.DoubleValue(value=max_vel)
    max_acc = wrappers_pb2.DoubleValue(value=max_acc)
    traj_list = []
    duration = 0
    for idx,joints in enumerate(joint_targets):
        sh0,sh1,el0,el1,wr0,wr1 = joints
        traj_point = RobotCommandBuilder.create_arm_joint_trajectory_point(
            sh0, sh1, el0, el1, wr0, wr1, times[idx])
        traj_list.append(traj_point)
        duration+= times[idx]
    print("Created traj list")
    arm_joint_traj = arm_command_pb2.ArmJointTrajectory(points=traj_list,
                                                        maximum_velocity=max_vel,
                                                        maximum_acceleration=max_acc)
    # Make a RobotCommand
    command = make_robot_command(arm_joint_traj)

    # Send the request
    cmd_id = clients.command.robot_command(command)
    block_until_arm_arrives(clients.command, cmd_id, duration+2.0)

    return cmd_id

def reachability(client: Clients, pose_frames, frame_name = GRAV_ALIGNED_BODY_FRAME_NAME):
    """
    Check reachibility of for the given pose when robot is in fixed standing pose
    Useful to decide when the robot cannot reach, hence allow robot to move closer
    True: Reachable
    False: Unreachable
    """
    body_T_task, wr1_T_tool,task_T_desired_tool =  pose_frames 

    ik_request = InverseKinematicsRequest(
                root_frame_name=frame_name,
                scene_tform_task=body_T_task.to_proto(),
                wrist_mounted_tool=InverseKinematicsRequest.WristMountedTool(
                    wrist_tform_tool=wr1_T_tool.to_proto()),
                tool_pose_task=InverseKinematicsRequest.ToolPoseTask(
                    task_tform_desired_tool=task_T_desired_tool.to_proto()),
            )
    
    ik_response = client.ik.inverse_kinematics(ik_request)
    reachable_ik = ik_response.status == InverseKinematicsResponse.STATUS_OK
    
    return reachable_ik

def move_impedance_control(clients: Clients, pose, wr1_T_tool = None, target_frame_name = BODY_FRAME_NAME, duration = 1.0):
    """
    Impedance control function
    """
    x,y,z,qw,qx,qy,qz = pose

    # wr1_T_tool = SE3Pose(x, y, z, rot=Quat(w=qw,x=qx,y=qy,z=qz))

    # If a nominal tool frame is not specified, default to using the hand frame.
    if wr1_T_tool is None:
        wr1_T_tool = get_a_tform_b(clients.state.get_robot_state().kinematic_state.transforms_snapshot,
                                       WR1_FRAME_NAME, HAND_FRAME_NAME)


    # Impedance command step
    robot_cmd = robot_command_pb2.RobotCommand()
    impedance_cmd = robot_cmd.synchronized_command.arm_command.arm_impedance_command
    impedance_cmd.root_frame_name = target_frame_name
    impedance_cmd.wrist_tform_tool.CopyFrom(wr1_T_tool.to_proto())
    impedance_cmd.diagonal_stiffness_matrix.CopyFrom(
        geometry_pb2.Vector(values=[10, 10, 10, 0.1, 0.1, 0.1]))
    impedance_cmd.diagonal_damping_matrix.CopyFrom(
        geometry_pb2.Vector(values=[0.5, 0.5, 0.5, 1.0, 1.0, 1.0]))

    # Set up our `desired_tool` trajectory. This is where we want the tool to be with respect
    # to the task frame. The stiffness we set will drag the tool towards `desired_tool`.
    traj = impedance_cmd.task_tform_desired_tool
    pt1 = traj.points.add()
    pt1.time_since_reference.CopyFrom(seconds_to_duration(duration))
    pt1.pose.CopyFrom(SE3Pose(x, y, z, Quat(qw, qx, qy, qz)).to_proto())
    pt2 = traj.points.add()
    pt2.time_since_reference.CopyFrom(seconds_to_duration(duration))
    pt2.pose.CopyFrom(SE3Pose(x, y, z, Quat(qw, qx, qy, qz)).to_proto())

    # Execute the impedance command.
    cmd_id = clients.command.robot_command(robot_cmd)

    return cmd_id

def get_current_gripper_pose(clients: Clients, target_frame_name = BODY_FRAME_NAME):
    """
    Get the homogeneous transformation matrix of the current hand pose frame w.r.t to
    the target frame
    """
    transforms = clients.state.get_robot_state().kinematic_state.transforms_snapshot
    # Transformation of hand w.r.t grav_aligned
    BODY_T_HAND = get_a_tform_b(transforms, target_frame_name, HAND_FRAME_NAME)
    # print(BODY_T_HAND)
    # xyz_rot = SE3Pose.from_matrix(BODY_T_HAND) # (x,y,z,quat)
    x, y, z = BODY_T_HAND.x, BODY_T_HAND.y, BODY_T_HAND.z
    qw, qx, qy, qz = BODY_T_HAND.rot.w, BODY_T_HAND.rot.x, BODY_T_HAND.rot.y, BODY_T_HAND.rot.z
    return BODY_T_HAND, (x, y, z, qw, qx, qy, qz)

# TODO: Move to utils
def print_feedback(feedback_resp, logger):
    """ Helper function to query for ArmJointMove feedback, and print it to the console.
        Returns the time_to_goal value reported in the feedback """
    
    joint_move_feedback = feedback_resp.feedback.synchronized_feedback.arm_command_feedback.arm_joint_move_feedback
    logger.info(f'  planner_status = {joint_move_feedback.planner_status}')
    logger.info(
        f'  time_to_goal = {duration_to_seconds(joint_move_feedback.time_to_goal):.2f} seconds.')

    # Query planned_points to determine target pose of arm
    logger.info('  planned_points:')
    for idx, points in enumerate(joint_move_feedback.planned_points):
        pos = points.position
        pos_str = f'sh0 = {pos.sh0.value:.3f}, sh1 = {pos.sh1.value:.3f}, el0 = {pos.el0.value:.3f}, ' \
                  f'el1 = {pos.el1.value:.3f}, wr0 = {pos.wr0.value:.3f}, wr1 = {pos.wr1.value:.3f}'
        logger.info(f'    {idx}: {pos_str}')
    return duration_to_seconds(joint_move_feedback.time_to_goal)

def get_camera_intrinsics(image_proto):
    """
    Get camera instrinsics
    """
    focal_x = image_proto.source.pinhole.intrinsics.focal_length.x
    principal_x = image_proto.source.pinhole.intrinsics.principal_point.x

    focal_y = image_proto.source.pinhole.intrinsics.focal_length.y
    principal_y = image_proto.source.pinhole.intrinsics.principal_point.y

    return [focal_x,focal_y, principal_x, principal_y]

def get_gpe_in_cam(rgb_res, clients: Clients):
    transforms_hand = rgb_res.shot.transforms_snapshot
    transforms_body = clients.state.get_robot_state().kinematic_state.transforms_snapshot
    odon_in_cam = get_a_tform_b(transforms_hand, rgb_res.shot.frame_name_image_sensor, VISION_FRAME_NAME)
    gpe_in_odom = get_a_tform_b(transforms_body, VISION_FRAME_NAME, GROUND_PLANE_FRAME_NAME)
    gpe_in_cam = odon_in_cam * gpe_in_odom
    return gpe_in_cam

