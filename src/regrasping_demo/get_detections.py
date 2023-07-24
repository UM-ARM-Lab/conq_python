import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rerun as rr
import matplotlib.pyplot as plt
from PIL import Image
from bosdyn.api import image_pb2, geometry_pb2, ray_cast_pb2
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import get_a_tform_b, BODY_FRAME_NAME, GROUND_PLANE_FRAME_NAME
from bosdyn.client.image import pixel_to_camera_space

from arm_segmentation.viz import viz_predictions
from conq.cameras_utils import get_color_img, get_depth_img, gpe_frame_in_cam
from conq.exceptions import DetectionError
from regrasping_demo.cdcpd_hose_state_predictor import single_frame_planar_cdcpd
from regrasping_demo.detect_regrasp_point import get_masks, detect_object_center, detect_regrasp_point_from_hose, \
    get_center_of_mass

DEFAULT_IDEAL_DIST_TO_OBS = 70


@dataclass
class GetRetryResult:
    image_res: image_pb2.ImageResponse
    hose_points: np.ndarray
    best_idx: int
    best_se2: geometry_pb2.SE2Pose
    best_vec2: geometry_pb2.Vec2


def np_to_vec2(a):
    return geometry_pb2.Vec2(x=a[0], y=a[1])


def vec3_to_np(v):
    return np.array([v.x, v.y, v.z])

def construct_se2_from_points(p_idx, body_pose_in_gpe, projected_points):
    pos = geometry_pb2.Vec2(x=projected_points[p_idx][0], y=projected_points[p_idx][1])
    
    # generate rotation from the rate of change 
    #if the selected point is not the first or last on the hose
    if p_idx == 0:
        p_n1 = projected_points[p_idx]
        p_n2 = projected_points[p_idx + 1]
    elif p_idx == len(projected_points) - 1:
        p_n1 = projected_points[p_idx - 1]
        p_n2 = projected_points[p_idx]
    else:
        # rotation is calculated using the two adjacent hose points
        p_n1 = projected_points[p_idx - 1]
        p_n2 = projected_points[p_idx + 1]

    # slope of line connecting neighbors
    m = (p_n2[1] - p_n1[1]) / (p_n2[0] - p_n1[0])

    # Construct a vector perpendicular to this line
    inv_m = -1.0 / m
    dir_vec = np.array([1, inv_m])
    dir_vec = dir_vec / np.linalg.norm(dir_vec)
    # find the angle between the vector and the x-axis
    angle = np.arctan(dir_vec[1] / dir_vec[0])

    # If start angle and the calculated pose are more than 90 degrees apart, flip the final pose
    start_to_goal_vec = np.array([projected_points[p_idx][0] - body_pose_in_gpe[0], projected_points[p_idx][1] - body_pose_in_gpe[1]])
    start_to_goal_vec = start_to_goal_vec / np.linalg.norm(dir_vec)
    if np.dot(dir_vec, start_to_goal_vec) < 0:
        angle = (angle + np.pi) % (2 * np.pi)
    return geometry_pb2.SE2Pose(position=pos, angle=angle)

def save_data(rgb_np, depth_np, predictions):
    now = int(time.time())
    Path(f"data/{now}").mkdir(exist_ok=True, parents=True)

    Image.fromarray(rgb_np).save(f"data/{now}/rgb.png")
    Image.fromarray(np.squeeze(depth_np)).save(f"data/{now}/depth.png")
    with open(f"data/{now}/pred.json", 'w') as f:
        json.dump(predictions, f)

def project_point(px, rgb_res, gpe_in_cam):
    '''
    Projects a point in camera frame into the ground plane
    Inputs:
        px: the pixel to be projected into ground plane
        rgb_res: a bosdyn image_pb2.ImageResponse from the image protobuf containing information about the rgb_np
        gpe_in_cam: a bosdyn math_helpers.SE3Pose representing the transform from the camera frame to the GPE frame

    Returns:
        point: the projected point in R3 space 
    '''
    p0 = np.array([gpe_in_cam.position.x, gpe_in_cam.position.y, gpe_in_cam.position.z])
    # 4x4 rotation matrix of gpe
    rot_mat_gpe = gpe_in_cam.rotation.to_matrix()
    plane_q = np.array([gpe_in_cam.rotation.x, gpe_in_cam.rotation.y, gpe_in_cam.rotation.z, gpe_in_cam.rotation.w])
    # normal vector of gpe, this is a numpy array
    n = rot_mat_gpe[0:3,2]

    l = np.array([*pixel_to_camera_space(rgb_res, px[0], px[1])])
    l = l / np.linalg.norm(l)
    l0 = np.array([0,0,0])

    d = np.dot((p0 - l0), n) / np.dot(l,n)
    point = l0 + l * d
    return point



def project_hose(predictor, rgb_np, rgb_res, gpe_in_cam):
    '''
    Projects the hose in an image into the ground plane.
    Inputs:
        rgb_np: an rgb image of the hose expressed as a numpy array.
        rgb_res: a bosdyn image_pb2.ImageResponse from the image protobuf containing information about the rgb_np
        gpe_in_cam: a bosdyn math_helpers.SE3Pose representing the transform from the camera frame to the GPE frame
    
    Returns:
        ordered_hose_points: the pixel space points on the hose that are to be projected 
        intersection: a numpy array containing 3D points of the hose projected onto the ground plane
    '''

    # Vec3 describing a point on the plane
    p0 = np.array([gpe_in_cam.position.x, gpe_in_cam.position.y, gpe_in_cam.position.z])
    # 4x4 rotation matrix of gpe
    rot_mat_gpe = gpe_in_cam.rotation.to_matrix()
    plane_q = np.array([gpe_in_cam.rotation.x, gpe_in_cam.rotation.y, gpe_in_cam.rotation.z, gpe_in_cam.rotation.w])
    # normal vector of gpe, this is a numpy array
    n = rot_mat_gpe[0:3,2]

    # get the prediction values for the hose
    predictions = predictor.predict(rgb_np)
    # n x 2 array with n points and their u, v pixel coordinates
    ordered_hose_points = single_frame_planar_cdcpd(rgb_np, predictions)

    head_px = detect_object_center(predictions, "vacuum_head")

    # project the vacuum head onto the ground plane
    hose_head_point = project_point(head_px, rgb_res, gpe_in_cam)

    rr.log_point('hose_head', hose_head_point)

    # rr.log_arrow("plane/n", p0, n)
    # rr.log_obb("plane/obb", position=p0, rotation_q=plane_q, half_size=[3.5, 3.5, 0.005], label="ground plane")
    
    # in pixel_to_camera_space, the z's are called 'depth'
    zs = np.ones_like(ordered_hose_points[:,0])
    l = np.array([*pixel_to_camera_space(rgb_res, ordered_hose_points[:,0], ordered_hose_points[:,1], depth=zs)])
    l = np.transpose(l)
    # l = np.column_stack((l, np.ones(ordered_hose_points[:,0].shape)))
    l = l / np.linalg.norm(l, axis=1, keepdims=True)
    l0 = np.array([0,0,0])

    d = np.dot((p0 - l0), n) / np.dot(l,n)
    intersection = l0 + l * d[:, np.newaxis]
    return ordered_hose_points, intersection, predictions



def get_mess(predictor, rc_client, image_client):
    rgb_np, rgb_res = get_color_img(image_client, 'hand_color_image')
    depth_np, depth_res = get_depth_img(image_client, 'hand_depth_in_hand_color_frame')

    predictions = predictor.predict(rgb_np)

    save_data(rgb_np, depth_np, predictions)

    mess_masks = get_masks(predictions, "mess")

    if len(mess_masks) == 0:
        raise DetectionError("No mess detected")

    if len(mess_masks) != 1:
        print(f"Error: expected 1 mess, got {len(mess_masks)}")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.imshow(rgb_np, alpha=0.5, zorder=0)
    ax.imshow(depth_np, alpha=0.5, zorder=1)
    viz_predictions(rgb_np, predictions, predictor.colors, fig, ax)
    fig.show()

    mess_px, mess_py = get_center_of_mass(mess_masks[0])

    # compute position in camera frame of the pixel given a fixed depth, which has the right direction in 3D
    # but may be the wrong length.
    mess_dir_in_cam = np.array(pixel_to_camera_space(rgb_res, mess_px, mess_py, 1))
    mess_dir_in_cam = mess_dir_in_cam / np.linalg.norm(mess_dir_in_cam)
    cam_in_body = get_a_tform_b(rgb_res.shot.transforms_snapshot, BODY_FRAME_NAME, rgb_res.shot.frame_name_image_sensor)
    mess_dir_in_body = vec3_to_np(cam_in_body.rot * math_helpers.Vec3(*mess_dir_in_cam))
    mess_origin_in_body = vec3_to_np(cam_in_body.position)

    # Ray-cast to the floor (or whatever is nearest based on the robot's perception of the world)
    response = rc_client.raycast(mess_origin_in_body, mess_dir_in_body,
                                 [ray_cast_pb2.RayIntersection.Type.TYPE_GROUND_PLANE],
                                 min_distance=1.0, frame_name=BODY_FRAME_NAME)

    if len(response.hits) == 0:
        raise DetectionError("No raycast hits")

    hit = response.hits[0]
    mess_x = hit.hit_position_in_hit_frame.x
    mess_y = hit.hit_position_in_hit_frame.y

    print(f"Mess detected at {mess_x:.2f}, {mess_y:.2f}")
    return mess_x, mess_y

# Modified to also return a best SE2
def get_hose_and_head_point(predictor, image_client, robot_state_client):
    rgb_np, rgb_res = get_color_img(image_client, 'hand_color_image')
    gpe_in_cam = gpe_frame_in_cam(robot_state_client, rgb_res)

    hose_points, projected_points_in_cam, predictions = project_hose(predictor, rgb_np, rgb_res, gpe_in_cam)
    head_px = detect_object_center(predictions, "vacuum_head")
    rr.log_line_strip("rope", projected_points_in_cam, stroke_width=0.02)

    fig, ax = plt.subplots()
    ax.imshow(rgb_np, zorder=0)
    ax.scatter(hose_points[:,0], hose_points[:, 1], c='yellow', zorder=2)

    transforms = robot_state_client.get_robot_state().kinematic_state.transforms_snapshot
    transforms_cam = rgb_res.shot.transforms_snapshot
    frame_name_shot = rgb_res.shot.frame_name_image_sensor
    projected_points_in_gpe = []
    for pt_in_cam in projected_points_in_cam:
        vec_in_cam = math_helpers.Vec3(*pt_in_cam)
        vec_in_gpe = get_a_tform_b(transforms, GROUND_PLANE_FRAME_NAME, BODY_FRAME_NAME) * get_a_tform_b(transforms_cam, BODY_FRAME_NAME, frame_name_shot) * vec_in_cam
        projected_points_in_gpe.append([vec_in_gpe.x, vec_in_gpe.y, vec_in_gpe.z])

    projected_points_in_gpe = np.array(projected_points_in_gpe)
    # TODO: Distances are measured in pixel space. Fix to be measured in R2
    dists = np.linalg.norm(hose_points - head_px, axis=-1)
    best_idx = int(np.argmin(dists))
    best_se2 = construct_se2_from_points(best_idx, projected_points_in_gpe)

    best_px = hose_points[best_idx]
    best_vec2 = np_to_vec2(best_px)

    return GetRetryResult(rgb_res, hose_points, best_idx, best_se2, best_vec2)

def get_hose_and_regrasp_point(predictor, image_client, robot_state_client, ideal_dist_to_obs=DEFAULT_IDEAL_DIST_TO_OBS):
    rgb_np, rgb_res = get_color_img(image_client, "hand_color_image")
    gpe_in_cam = gpe_frame_in_cam(robot_state_client, rgb_res)
    hose_points, projected_points, predictions = project_hose(predictor, rgb_np, rgb_res, gpe_in_cam)
    
    best_idx, best_px = detect_regrasp_point_from_hose(predictions, hose_points, ideal_dist_to_obs)
    best_se2 = construct_se2_from_points(best_idx, projected_points)
    best_vec2 = np_to_vec2(best_px)

    return GetRetryResult(rgb_res, hose_points, best_idx, best_se2, best_vec2)

