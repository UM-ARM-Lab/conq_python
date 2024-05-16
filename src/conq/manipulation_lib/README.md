# Manipulation Module Docs

## Manipulation Libraries Overview
- **`Manipulation.py`**: Contains the basic manipulation functions that are basically wrapper functions over BOSDYN API. Commonly used functions are: *`move_gripper()`, `open_gripper()`, `close_gripper()`, `move_joint_trajectory()`, `move_impedance_control()`*
- **`Grasp.py`**: Runs a subprocess that runs the docker container to get grasp inference from GPD (Grasp Pose Detection). Requires docker installation
- **`Perception3D.py`**: Contains *`PointCloud`, `Vision` and `VisualPoseAcquirer`* classes
- **`VisualServo`**: PD controller for PBVS. Requires async motion and vision acquisition

## GPD Docker Installation Steps
#### STEP 0: Docker installed with sudo rights:
Refer https://docs.docker.com/engine/install/ubuntu/ for the installation steps. **Don't use apt-get to install docker!!!**

Make sure to have docker installed with sudo rights if the installation didn't take care of it. 
```shell
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker
```
You should be able to run `docker run hello-world` without sudo after you run these commands

Refer https://docs.docker.com/engine/install/linux-postinstall/ if you face additional issues.
#### STEP 1: Clone the modified GPD repository:
cd to `src/conq/manipulation_lib`
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

**Note:** Recent commits to the PCL library may result in the docker image being built. Then load the docker image instead of building it.

#### (ALTERNATIVE) Load the docker image

If you have access to the `conq_gpd.tar` image file, then do:
```shell
docker load < conq_gpd.tar
```

#### STEP 3: Run the container and build the files

```shell
docker run -it \
-v /home/$user/path/to/repo/conq_python/src/conq/manipulation_lib/gpd:/gpd \
-v /tmp/.X11-unix:/tmp/.X11-unix \
conq_gpd:<instance_name> \
cd gpd && cd build && cmake .. && make -j$(nproc) \
make install
exit
```

Make sure to replace `$user`, `path` and the docker container `<instance_name>` with appropriate values.

#### STEP 4: Example usage check:
Have a sample.pcd file under `data/PCD/`. Once you are done with the previous steps, run the following to check if everything works.

```shell
docker run -it \
-v /home/$user/path/to/repo/conq_python/src/conq/manipulation_lib/gpd:/gpd \
-v /tmp/.X11-unix:/tmp/.X11-unix \
conq_gpd:<instance_name> \
/bin/bash -c cd gpd/build && ./detect_grasps ../cfg/eigen_params.cfg ../data/PCD/<file>.pcd
exit
```

#### PYTHON check:
If you have completed the previous steps, you may run the python script that is basically a wrapper function to get the grasp poses by running the container as a submodule.

Run the `Grasp.py` file under `src/conq/manipulation_lib/`
```python3
python3 Grasp.py
```

#### FUTURE DIRECTIONS:
##### Parameters to tune
Since GPD performs grasp candidate sampling based on geometry at the early stages, we need to make sure that the configurations match the hand geometry. These can be changed under `gpd/cfg/eigen_params.cfg`. Make sure it reflects the gripper geometry being used. After modifications, you'll have to build the files again. Sometimes, deleting the contents under build files might help navigating issues.
##### GPU Usage
As of now, GPD runs in CPU. Depending on the CPU, it takes about 1-3 seconds for inference for 3 grasp candidates. The official repository has instructions to use GPD with GPU https://github.com/atenpas/gpd
