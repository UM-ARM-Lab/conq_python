# Arm Pose
Script that allows you to control the arm through the python api (using OpenCV sliders)
and prints the orientations to the screen for ease of use

Based on arm_simple.py example

## Estop

You will need to launch a software e-stop separately. The E-Stop programming example is [here](../estop/README.md).
You can also set the tablet to be the Estop:
On the tablet, press "power"->'Advanced'->'Release Control' (The tablet will maintain estop control, but not command control). You can also run the estop example if you want estop control on the laptop.

On your laptop, connect to the robot's wifi network and login using the credentials on the sticker of the battery housing. Running the commands below will automatically claim command control

Spot Login info is on a sticker on the battery housing.
Admin username: admin
User Username: user
Core Username: spot
Core Password: <lab password>
Spot's IP address: 192.168.80.3

It is often convenient to run (with the correct password): 
```
export BOSDYN_CLIENT_USERNAME=user && export ROBOT_IP=192.168.80.3 && export BOSDYN_CLIENT_PASSWORD=
```

## Running the Script
From within the scripts/print_arm_pose folder, run:

```
python3 print_arm_pose.py 192.168.80.3
```
