import re
import subprocess
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import pdist
import open3d as o3d
import os

REPO_DIR = os.getcwd()

"""
EXAMPLE USAGE:
docker run -it -e DISPLAY -v <REPO_DIR>/src/conq/manipulation_lib/gpd:/gpd -v /tmp/.X11-unix:/tmp/.X11-unix conq_gpd:stanley
"""

def get_grasp_candidates(file="live"):
    print("Starting Docker container...")
    docker_run_command = [
        "docker", "run", "-it",
        "-v", REPO_DIR+"/src/conq/manipulation_lib/gpd:/gpd",
        "conq_gpd:stanley",
        "/bin/bash",
        "-c",
        f"cd gpd/build && ./detect_grasps ../cfg/eigen_params.cfg ../data/PCD/{file}.pcd",
        "exit",
    ]

    # Run the Docker command and capture output
    process = subprocess.Popen(
        docker_run_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()

    # Combine stdout and stderr into a single string
    output_text = stdout.decode() + stderr.decode()

    # expression pattern to extract grasp information
    pattern = r"Grasp #(\d+): Score = (-?\d+\.\d+).*?Position: \[(.*?)\].*?Orientation \(Quaternion\): \[w: (-?\d+\.\d+), x: (-?\d+\.\d+), y: (-?\d+\.\d+), z: (-?\d+\.\d+)\]"

    # Extract grasp information using regex
    grasp_candidates = []
    for match in re.finditer(pattern, output_text, re.DOTALL):
        number = int(match.group(1))
        score = float(match.group(2))
        position = list(map(float, match.group(3).split(", ")))
        orientation = {
            "w": float(match.group(4)),
            "x": float(match.group(5)),
            "y": float(match.group(6)),
            "z": float(match.group(7)),
        }
        grasp_candidate = {
            "number": number,
            "score": score,
            "position": position,
            "orientation": orientation,
        }
        grasp_candidates.append(grasp_candidate)
    #print("Got grasp candidates: ", grasp_candidates)
    return grasp_candidates


def get_best_grasp_pose(Target_T_Source, min_grass_z=None, file="live", to_body=False):
    "Returns best grasp pose as tuple"
    grasp_candidates = get_grasp_candidates(file) # Grasp candidates from hand_sensor_frame
    
    grasp_cand_list = []
    grasp_pos_list = []
    grasp_quat_list = []

    # print("Grasps in hand frame:")
    # for grasp in grasp_candidates:
    #     print(grasp)
    # print()

    for grasp in grasp_candidates:
        
        pos = np.array(grasp["position"]) #
        quat = np.array(tuple(dict_to_tuple_wxyz(grasp["orientation"]))) # [qw, qx, qy, qz]

        grasp_pos_list.append(pos)
        grasp_quat_list.append(quat)
        pose_print = transform_grasp_pose(grasp,Target_T_Source)
        grasp_cand_list.append(pose_print)

    grasp_cand_array = np.hstack((grasp_pos_list,grasp_quat_list)) # nd.array: N x 7 (x,y,z,qw,qx,qy,qz)
    print(f"Found total {grasp_cand_array.shape[0]} grasp candidates.")
    # Visualize grasp candidates in hand pose frame
    # Path to the point cloud file
    PCD_PATH = "src/conq/manipulation_lib/gpd/data/PCD/live.pcd"
    # Load the point cloud
    point_cloud = o3d.io.read_point_cloud(PCD_PATH)
    # viz_grasp_cand(point_cloud, grasp_cand_array)
    
    if to_body:
        print("Grasp from target frame")
        position = tuple(grasp_candidates[0]["position"])
        orientation = tuple(dict_to_tuple_scipy(grasp_candidates[0]["orientation"])) #(qw, qx, qy, qz)
        pose_tuple = position + orientation
    else:
        # pose_tuple = transform_grasp_pose(grasp_candidates[0],Target_T_Source)
        # print("Final pose",pose_tuple)
        gaze_pose = (0.75, 0.0, -0.1, 0.7071, 0., 0.7071, 0.)
        pose_tuple = choose_best_grasp(grasp_cand_list, gaze_pose, min_grass_z=min_grass_z)
    return pose_tuple

def transform_grasp_pose(grasp_candidate,Target_T_Source, raw_grasp_pose=False):
    "Transform grasp pose from sensor frame to Body frame given transformation matrix"

    position = None
    rot = None

    if raw_grasp_pose:
        position = grasp_candidate[:3]
        quat = grasp_candidate[3:]
        rot = R.from_quat(quat).as_matrix() # 3 x 3
    else:
        position = np.array(grasp_candidate["position"])
        quat = list(tuple(dict_to_tuple_scipy(grasp_candidate["orientation"]))) # [qx, qy, qz, qw]
        # convert quat to rot
        rot = R.from_quat(quat).as_matrix() # 3 x 3

    Hand_T_Grasp = np.eye(4)
    Hand_T_Grasp[:3, :3] = rot
    Hand_T_Grasp[:3, 3] = position

    # Convert from Hand vision to Body
    pose = np.dot(Target_T_Source,Hand_T_Grasp) # 4 x 4
    position,rot = pose[:3, 3], pose[:3, :3]
    rot = R.from_matrix(rot)
    quat = rot.as_quat() # (qx, qy, qz, qw)
    quat = [quat[3], quat[0], quat[1], quat[2]] #(qw, qx, qy, qz)
    pose_tuple = tuple(round(elem, 4) for elem in position.tolist() + quat)

    return pose_tuple

def transform_pose_body_to_hand(grasp_pose_body, Body_T_Hand, raw_grasp_pose=False):
    """
    Transform grasp pose from Body frame to Hand frame given the transformation matrix.
    
    :param grasp_pose_body: The grasp pose in the body frame (either raw or as a dictionary)
    :param Body_T_Hand: The transformation matrix from the Body frame to the Hand frame (inverse of Target_T_Source)
    :param raw_grasp_pose: Boolean indicating if the grasp pose is provided as raw values (True) or as a dictionary (False)
    :return: Transformed grasp pose in the hand frame as a tuple
    """
    position = None
    rot = None

    if raw_grasp_pose:
        position = grasp_pose_body[:3]
        quat = grasp_pose_body[3:]
        rot = R.from_quat(quat).as_matrix()  # 3 x 3
    else:
        position = np.array(grasp_pose_body["position"])
        quat = list(tuple(dict_to_tuple_scipy(grasp_pose_body["orientation"])))  # [qx, qy, qz, qw]
        rot = R.from_quat(quat).as_matrix()  # 3 x 3

    Body_T_Grasp = np.eye(4)
    Body_T_Grasp[:3, :3] = rot
    Body_T_Grasp[:3, 3] = position

    # Convert from Body to Hand
    pose = np.dot(Body_T_Hand, Body_T_Grasp)  # 4 x 4
    position, rot = pose[:3, 3], pose[:3, :3]
    rot = R.from_matrix(rot)
    quat = rot.as_quat()  # (qx, qy, qz, qw)
    quat = [quat[3], quat[0], quat[1], quat[2]]  # (qw, qx, qy, qz)
    pose_tuple = tuple(round(elem, 4) for elem in position.tolist() + quat)

    return pose_tuple

def get_object_width_at_grasp(grasp_pose_body, Body_T_Hand):

    PCD_PATH = "src/conq/manipulation_lib/gpd/data/PCD/live.pcd"

    pcd = o3d.io.read_point_cloud(PCD_PATH)
    points = np.asarray(pcd.points)

    #grasp pose in robot hand frame (x, y, z, qw, qx, qy, qz)
    Hand_T_Body = np.linalg.inv(Body_T_Hand)
    grasp_pose_hand = transform_grasp_pose(list(grasp_pose_body), Hand_T_Body, raw_grasp_pose=True)
    grasp_pose = np.array(grasp_pose_hand)

    #convert quaternion to rotation matrix
    rotation_matrix = R.from_quat(grasp_pose[3:]).as_matrix()

    # extract z axis from grasp pose
    z_axis = rotation_matrix[:, 2]

    # step size and a range for searching along the z axis
    step_size = .001  # in meters
    num_steps = 70  #number of steps up and down

    collected_points = []
    z_axis_points = []

    for i in range(-num_steps, num_steps):
        point_on_z = grasp_pose[:3] + i * step_size * z_axis
        z_axis_points.append(point_on_z)
        
        # Compute distances to all points in the cloud
        distances = np.linalg.norm(points - point_on_z, axis=1)
        
        # Threshold to determine if a point is on the Z-axis
        threshold = 0.015  # in meters
        mask = distances < threshold
        
        # Collect points that are close to the current Z-axis position
        collected_points.extend(points[mask])

    collected_points = np.array(collected_points)
    z_axis_points = np.array(z_axis_points)


    local_object_width = 1
    if len(collected_points) < 2:
        print("Not enough points were collected to compute the distance.")
    else:
        # visualize pcd, z axis points, and local grasp points
        collected_pcd = o3d.geometry.PointCloud()
        collected_pcd.points = o3d.utility.Vector3dVector(collected_points)
        collected_pcd.paint_uniform_color([0, 1, 0])  # Green for collected points
        z_axis_pcd = o3d.geometry.PointCloud()
        z_axis_pcd.points = o3d.utility.Vector3dVector(z_axis_points)
        z_axis_pcd.paint_uniform_color([0, 0, 1])  # Blue for Z-axis points
        pcd.paint_uniform_color([1, 0, 0])  # Red
        o3d.visualization.draw_geometries([pcd, z_axis_pcd, collected_pcd])
        
        pairwise_distances = pdist(collected_points) #compute the pairwise distances between all collected points
        local_object_width = np.max(pairwise_distances) #get maximum distance, which is the estimated width of the grasp location
        print(f"The local object width is: {local_object_width} meters")

    return local_object_width

def grasp_width_to_gripper_open_percent(object_width):

    open_percent = 556.0617 * object_width - 3.7339 # obtained from linear regression of width vs open percent

    if open_percent > 100.0:
        open_percent = 100.0
    elif open_percent < 6.0:
        open_percent = 0.0

    return open_percent


def quaternion_dot(q1, q2):
    return np.dot(q1, q2)

#Either filter grasps by approach axis or Z-height
def choose_best_grasp(grasp_candidates, gaze_pose, min_grass_z=None):
    # Extract the quaternion part of the gaze pose
    gaze_quaternion = gaze_pose[3:]
    
    best_grasp = None
    best_similarity = -1  # Start with the lowest possible dot product value
    
    for grasp in grasp_candidates:
        grasp_quaternion = grasp[3:]

        if min_grass_z is not None and grasp[2] < min_grass_z:
            print(f'Eliminated low grasp: {grasp}')
            continue
        else:

            #filter by approach axis
            similarity = quaternion_dot(gaze_quaternion, grasp_quaternion)

            #filter by choosing highest z value
            # similarity = grasp[2]
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_grasp = grasp
    return best_grasp

def dict_to_tuple_wxyz(orientation_dict):

    return (
        orientation_dict["w"],
        orientation_dict["x"],
        orientation_dict["y"],
        orientation_dict["z"],
    )
def dict_to_tuple_scipy(orientation_dict):

    return (
        orientation_dict["x"],
        orientation_dict["y"],
        orientation_dict["z"],
        orientation_dict["w"],
    )

def viz_grasp_cand(point_cloud, grasp_candidates_array):
    geometries = [point_cloud]

    # coordinate frame for the origin
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25, origin=[0, 0, 0])
    geometries.append(coordinate_frame)

    grasp_points = grasp_candidates_array[:,:3]
    grasp_orientations = grasp_candidates_array[:,3:]

    for i, (point, orientation) in enumerate(zip(grasp_points, grasp_orientations)):
        # small sphere at the grasp point
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
        sphere.translate(point)
        sphere.paint_uniform_color([0, 0, 0])  # black color
        geometries.append(sphere)
        
        # coordinate frame at the grasp point with the given orientation
        grasp_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=point)
        
        # Convert quaternion to rotation matrix
        R = o3d.geometry.get_rotation_matrix_from_quaternion(orientation)
        grasp_frame.rotate(R, center=point)
        geometries.append(grasp_frame)

    # Visualize the point cloud along with the grasp points and their orientations
    o3d.visualization.draw_geometries(geometries)

def compute_grass_z_range(point_cloud_file):
    # Load the point cloud
    pcd = o3d.io.read_point_cloud(point_cloud_file)

    # Segment the largest plane in the point cloud using RANSAC
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.015, ransac_n=5, num_iterations=1000)

    [a, b, c, d] = plane_model
    # print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    # Extract inlier points (the plane)
    inlier_cloud = pcd.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])  # Paint the plane points red

    # Extract outlier points (the rest of the point cloud)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)

    # Find the inlier point with the highest Z value
    inlier_points = np.asarray(inlier_cloud.points)

    min_z_index = np.argmin(inlier_points[:, 2])
    min_z_point = inlier_points[min_z_index]

    max_z_index = np.argmax(inlier_points[:, 2])
    max_z_point = inlier_points[max_z_index]

    print(f"The grass point with the min Z value in hand frame is: {min_z_point}")
    print(f"The grass point with the max Z value in hand frame is: {max_z_point}")

    # Visualize the original point cloud with the fitted plane
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

    return min_z_point, max_z_point


def main():
    # Example usage   
    Target_T_Source = np.array([[-0.12314594,  0.99110129, -0.05053028,  0.052     ],
 [-0.57408013, -0.02961206,  0.81826349 , 0.117     ],
 [ 0.80948569,  0.12977425,  0.57261816,  0.701 ,    ],
 [ 0. ,         0.    ,      0.     ,     1.        ]])
    grasp_pose = get_best_grasp_pose(Target_T_Source,file="live",to_body=False)
    print(grasp_pose)

if __name__ == "__main__":
    main()
