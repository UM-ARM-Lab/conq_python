import subprocess
import re

"""
sudo docker run -it --rm -e DISPLAY -v /home/conq/Spot590/conq_python/src/conq/manipulation_lib/gpd:/gpd -v /tmp/.X11-unix:/tmp/.X11-unix conq_gpd:stanley
"""

def get_grasp_candidates(file="live"):

    docker_run_command = [
        "docker", "run", "-it", "--rm",
        "-e", "DISPLAY",
        "-v", "/home/conq/Spot590/conq_python/src/conq/manipulation_lib/gpd:/gpd",
        "-v", "/tmp/.X11-unix:/tmp/.X11-unix",
        "conq_gpd:stanley",
        "/bin/bash", "-c", f"cd gpd/build && ./detect_grasps ../cfg/eigen_params.cfg ../data/PCD/{file}.pcd"
    ]

    # Run the Docker command and capture output
    process = subprocess.Popen(docker_run_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    # Combine stdout and stderr into a single string
    output_text = stdout.decode() + stderr.decode()

    # expression pattern to extract grasp information
    pattern = r'Grasp #(\d+): Score = (-?\d+\.\d+).*?Position: \[(.*?)\].*?Orientation \(Quaternion\): \[w: (-?\d+\.\d+), x: (-?\d+\.\d+), y: (-?\d+\.\d+), z: (-?\d+\.\d+)\]'

    # Extract grasp information using regex
    grasp_candidates = []
    for match in re.finditer(pattern, output_text, re.DOTALL):
        number = int(match.group(1))
        score = float(match.group(2))
        position = list(map(float, match.group(3).split(', ')))
        orientation = {
            "w": float(match.group(4)),
            "x": float(match.group(5)),
            "y": float(match.group(6)),
            "z": float(match.group(7))
        }
        grasp_candidate = {
            "number": number,
            "score": score,
            "position": position,
            "orientation": orientation
        }
        grasp_candidates.append(grasp_candidate)

    return grasp_candidates

def get_best_grasp_pose(file="live"):
    "Returns best grasp pose as tuple"
    grasp_candidates = get_grasp_candidates(file)

    for grasp in grasp_candidates:
        print(grasp)

    position = tuple(grasp_candidates[0]["position"])
    orientation = tuple(dict_to_tuple(grasp_candidates[0]["orientation"]))

    return position+orientation

def dict_to_tuple(orientation_dict):
    
    return (orientation_dict["w"], orientation_dict["x"], orientation_dict["y"], orientation_dict["z"])

def main():
    file= "test_filtered" # TODO: Replace with "live" when the pointcloud is filtered and not empty
    grasp_pose = get_best_grasp_pose(file)
    print(grasp_pose)
    

if __name__ == '__main__':
    main()