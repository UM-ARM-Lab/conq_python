1. Login to the tablet as admin and establish estop control

2. On the tablet, press "power"->'Advanced'->'Release Control' (The tablet will maintain estop control, but not command control). You can also run the estop example if you want estop control on the laptop.

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

Go into the src/clickmap_nav folder. then running the tool_retrieval_interface (a descendent of ClickMapInterface) might look like:
```
python3 -m tool_retrieval_interface -a -u ~/spot/maps/collabspace3.walk --model-filepath ~/spot/models/garden_implements_model.pth 192.168.80.3
```
```
python3 -m tool_retrieval_interface -a -u ~/spot/maps/dude_design_studio1.walk --model-filepath ~/spot/models/model2.pth 192.168.80.3
```

1. You must Initialize the robot to the correct waypoint or fiducial (see controls below) before being able to navigate. Note that the robot must be in the exact position AND ORIENTATION of the waypoint. To do this, put your mouse over the desired waypoint on the map and then press key (4) to activate the initialization command.
2. To navigate, move your mouse over the desired waypoint and press key (g),(h),(j),or (k) to pick up the desired object.
