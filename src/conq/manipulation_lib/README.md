# Manipulation Module Docs

## Manipulation Libraries Overview
- **`Manipulation.py`**: Contains the basic manipulation functions that are basically wrapper functions over BOSDYN API. Commonly used functions are: *`move_gripper()`, `open_gripper()`, `close_gripper()`, `move_joint_trajectory()`, `move_impedance_control()`*
- **`Grasp.py`**: Runs a subprocess that runs the docker container to get grasp inference from GPD (Grasp Pose Detection). Requires docker installation
- **`Perception3D.py`**: Contains *`PointCloud`, `Vision` and `VisualPoseAcquirer`* classes
- **`VisualServo`**: PD controller for PBVS. Requires async motion and vision acquisition

## GPD Docker Installation Steps
Make sure to have docker installed with sudo rights.
#### STEP 1: Clone the modified GPD repository:
```shell
git clone https://github.com/JemuelStanley47/gpd.git 
cd gpd && mkdir build data
cd data && mkdir PCD RGB MASK DEPTH NPY
```

#### STEP 2: Build the dockerfile
```shell
docker build -t conq_gpd .
```
We will be running the container with a shared volume. So the live data will be stored under the directories: `gpd/data/<PCD/RGB/MASK/NPY/DEPTH>`

#### STEP 3: Run the container and build the files
```shell
docker run -it --rm \
-e DISPLAY \
-v /home/$user/path/to/repo/conq_python/src/conq/manipulation_lib/gpd:/gpd \
-v /tmp/.X11-unix:/tmp/.X11-unix \
conq_gpd:<instance_name> \
cd gpd && cd build && cmake .. && make -j$(nproc) \
make install
```

Make sure to add `$user`, `path` and the docker container `<instance_name>` accordingly.
#### STEP 4: Example usage check:
Have a sample.pcd file under `data/PCD/`. Once you are done with the previous steps, run the following to check if everything works.

```shell
docker run -it --rm \
-e DISPLAY \
-v /home/$user/path/to/repo/conq_python/src/conq/manipulation_lib/gpd:/gpd \
-v /tmp/.X11-unix:/tmp/.X11-unix \
conq_gpd:<instance_name> \
/bin/bash -c cd gpd/build && ./detect_grasps ../cfg/eigen_params.cfg ../data/PCD/<file>.pcd
exit
```
#### OR
Run the `Grasp.py` file under `src/conq/manipulation_lib/`
```python3
python3 Grasp.py
```

#### FUTURE DIRECTIONS:
##### Parameters to tune
##### GPU Usage
