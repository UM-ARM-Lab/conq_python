# Clickmap_nav: 
This is a combination of the view_map and graph_nav examples. You can use it to view a map that spot has made, and then click on the map to send movement commands.

## Setup Dependencies

This example requires VTK (visualization toolkit) and Numpy, and requires python 3. Using pip, these dependencies can be installed using:

```
python3 -m pip install -r requirements.txt
```
## Preliminaries

New users should follow the Quickstart Guide (https://dev.bostondynamics.com/docs/python/quickstart#verify-you-can-command-and-query-spot) to get spot up and running. 

1. Record a map using the tablet's AutoWalk or the Command Line interface. (If using Autowalk, transfer the map from Documents/bosdyn/autowalk/your_map.walk to your local machine using a USB cable). The map should be a directory of the form:

```
- /your_map.walk
    + graph
    - waypoint_snapshots
    - edge_snapshots
```

Always remove the battery after use!

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

## Running the click-Map Interface 
From within the clickmap_nav folder, run:
```
python3 -m click_map_interface -a -u <path/to/map/directory> 192.168.80.3
```
-a refers to anchoring. You can leave off the -a if your map doesn't have anchoring or if you're ok with a messier map.

-u is the flag to indicate the upload filepath.

A full example might look like: 
```
python3 -m click_map_interface -a -u ~/spot/maps/collabspace3.walk 192.168.80.3
```

Running the tool_retrieval_interface (a descendent of ClickMapInterface) might look like:
```
python3 -m tool_retrieval_interface -a -u ~/spot/maps/collabspace3.walk --model-filepath ~/spot/models/garden_implements_model.pth 192.168.80.3
```


Note: If you're already running the estop example on the laptop then you may not need the IP address at the end

Note: to avoid having to log in each time, run:
```
export BOSDYN_CLIENT_USERNAME=user && export BOSDYN_CLIENT_PASSWORD=<password_on_sticker>
```

## Using the Click-map Interface
1. You must Initialize the robot to the correct waypoint or fiducial (see controls below) before being able to navigate. Note that the robot must be in the exact position AND ORIENTATION of the waypoint. To do this, put your mouse over the desired waypoint on the map and then press key (4) to activate the initialization command.
2. To navigate, move your mouse over the desired waypoint and press key (6) to activate the "Navigate To" command. Note that it may take a few seconds to respond.

## Controls
From VTK:
(Right-Click)  Zoom
(Left-Click)   Rotate
(Scroll-Click) Pan
(r) reset the camera
(e) exit the program
(f) set a new camera focal point and fly towards that point
(u) invokes the user event
(3) toggles between stereo and non-stero mode
(l) toggles on/off a latitude/longitude markers that can be used to estimate/control position.

From Boston Dynamics (mostly the same as examples)
(1) Get localization state.
(2) Initialize localization to the nearest fiducial (must be in sight of a fiducial).
(4) Initialize localization to a specific waypoint (must be exactly at the waypoint).
(5) (Re)Upload the graph and its snapshots.
(6) Navigate to. The destination waypoint id is the second argument.
(8) List the waypoint ids and edge ids of the map on the robot.
(9) Clear the current graph.
(q) Exit.

## Testing Subcomponents
1. Run the map viewer alone
```
python3 -m view_map_with_highlight -a <path_to_your_map_directory>
```
For example: 
```
python3 -m view_map_highlighted -a ~/spot/maps/collabspace3.walk
```

2. Run command_line_interface.py alone (same behavior as the graph_nav_command_line example)

## Code Structure
graph_nav_interface.py contains a base class with all of the functionality to interact with boston dynamics' graph_nav service, and also maintain the robot lease etc.

click_map_interface.py contains a class that allows you to input commands to the GraphNavInterface base through a visual map. 

command_line_interface.py contains a class that allows you to input commands to the GraphNavInterface base through the command-line. When run, this has the same functionality as the graph_nav_command_line example in the SDK.

view_map_highlighted.py contains classes that make it easier for the Boston Dynamics SDK to work with VTK. For example, the VTKEngine class has all of the standard components necessary to render a map, and the BosdynVTKInterface class has many functions for creating the visual representations (ie. actors) corresponding to each type of map object (ie. waypoint, edge, fiducial, etc). When run, the functionality is similar to the view_map sdk example, but this also includes the ability to highlight actors with a silhouette, and the code is structured into more manageable classes.

bosdyn_vtk_utils.py contains helper functions that link VTK with Boston Dynamics (used for view_map_with_highlighted.py)

graph_nav_util.py contains helper functions for graph_nav_interface.py

## GraphNav Map Structure

GraphNav maps consist of waypoints, and edges between the waypoints. A waypoint consists of a reference frame, a name, a unique ID, and associated raw data. The raw data for a waypoint is stored in what is called a "Snapshot". Multiple waypoints may share the same snapshot.

Raw data includes feature clouds, April tag detections, imagery, terrain maps, etc.

Edges consist of a directed edge from one waypoint to another and a transform that estimates the relationship in 3D space between the two waypoints.

Maps do not have a global coordinate system (like GPS coordinates, for example). Only the relative transformations between waypoints are known.

## Understanding the Map Viewer

The map viewer displays waypoints as axes (red, green and blue arrows) where the "z" axis is blue, the "x" is red, and the "y" is green. They are connected by white lines representing edges.

Around the waypoints, the map viewer displays feature clouds. Feature clouds are collections of points that correspond to detected edge features in the robot's cameras. The feature clouds are colored by height, where blue is higher and red is lower.

The viewer also shows april tag detections as blue squares labeled with the fiducial ID. If multiple fiducials with the same ID are displayed near each other, this represents multiple detections taken at different times.
