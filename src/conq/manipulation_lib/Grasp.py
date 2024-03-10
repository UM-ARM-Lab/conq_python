"""
docker run -it --rm \
  -e DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /home/vision/Spot590/conq_python/data:/gpd/data \
  localhost/gpq_image:latest

"""
import subprocess

# Define the Docker run command
docker_run_command = [
    "docker", "run", "-it", "--rm",
    "-e", "DISPLAY",
    "-v", "/tmp/.X11-unix:/tmp/.X11-unix",
    "-v", "/home/vision/Spot590/conq_python/data:/gpd/data",
    # "-v", "/home/vision/Spot590/conq_python/src/conq/manipulation_lib/gpd:/gpd",
    "localhost/gpq_image:latest",
    "/bin/bash", "-c", "cd gpd/build && ./detect_grasps ../cfg/eigen_params.cfg ../data/PCD/krylon.pcd"
]

# Execute the Docker run command
subprocess.run(docker_run_command)
