import re
import subprocess
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
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


def get_best_grasp_pose(Target_T_Source, file="live", to_body=False):
    "Returns best grasp pose as tuple"
    grasp_candidates = get_grasp_candidates(file) # Grasp candidates from hand_sensor_frame
    
    grasp_cand_list = []
    grasp_pos_list = []
    grasp_quat_list = []
    for grasp in grasp_candidates:
        
        pos = np.array(grasp["position"]) #
        quat = np.array(tuple(dict_to_tuple_wxyz(grasp["orientation"]))) # [qw, qx, qy, qz]

        grasp_pos_list.append(pos)
        grasp_quat_list.append(quat)

        pose_print = transform_grasp_pose(grasp,Target_T_Source)
        grasp_cand_list.append(pose_print)

    grasp_cand_array = np.hstack((grasp_pos_list,grasp_quat_list)) # nd.array: N x 7 (x,y,z,qw,qx,qy,qz)
    # Visualize grasp candidates in hand pose frame
    # Path to the point cloud file
    PCD_PATH = "src/conq/manipulation_lib/gpd/data/PCD/live.pcd"
    # Load the point cloud
    point_cloud = o3d.io.read_point_cloud(PCD_PATH)
    viz_grasp_cand(point_cloud, grasp_cand_array)
    
    if to_body:
        print("Grasp from target frame")
        position = tuple(grasp_candidates[0]["position"])
        orientation = tuple(dict_to_tuple_scipy(grasp_candidates[0]["orientation"])) #(qw, qx, qy, qz)
        pose_tuple = position + orientation
    else:
        # pose_tuple = transform_grasp_pose(grasp_candidates[0],Target_T_Source)
        # print("Final pose",pose_tuple)
        gaze_pose = (0.75, 0.0, 0.3, 0.7071, 0., 0.7071, 0.)
        pose_tuple = choose_best_grasp(grasp_cand_list, gaze_pose)
    return pose_tuple

def transform_grasp_pose(grasp_candidate,Target_T_Source):
    "Transform grasp pose from sensor frame to Body frame given transformation matrix"
    position = np.array(grasp_candidate["position"]) #
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

def quaternion_dot(q1, q2):
    return np.dot(q1, q2)

def choose_best_grasp(grasp_candidates, gaze_pose):
    # Extract the quaternion part of the gaze pose
    gaze_quaternion = gaze_pose[3:]
    
    best_grasp = None
    best_similarity = -1  # Start with the lowest possible dot product value
    
    for grasp in grasp_candidates:
        grasp_quaternion = grasp[3:]
        similarity = quaternion_dot(gaze_quaternion, grasp_quaternion)
        
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
