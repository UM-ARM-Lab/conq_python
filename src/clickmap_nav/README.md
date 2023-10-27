# Clickmap_nav: 
This is a combination of the view_map and graph_nav examples. You can use it to view a map that spot has made, and then click on the map to send movement commands.

## Setup Dependencies

This example requires VTK (visualization toolkit) and Numpy, and requires python 3. Using pip, these dependencies can be installed using:

```
python3 -m pip install -r requirements.txt
```

## Running the Example

1. Record a map using AutoWalk or the Command Line interface. (If using Autowalk, transfer the map from Documents/bosdyn/autowalk/your_map.walk to your local machine using a USB cable). The map should be a directory of the form:

```
- /your_map.walk
    + graph
    - waypoint_snapshots
    - edge_snapshots
```
2. Run the click-map interface
```
python3 -m click_map_graph_nav_interface -a <path_to_your_map_directory>
```

## Testing Subcomponents
1. Run the map viewer alone
```
python3 -m view_map_with_highlight -a <path_to_your_map_directory>
```
For example: 
```
python3 -m view_map_with_highlight -a ~/spot/maps/collabspace3.walk
```
Note: -a is for anchoring. You can leave off the -a if your map doesn't have anchoring or if you're ok with a messier map.

2. Run command_line_graph_nav_interface.py alone 

## Camera Controls
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

## GraphNav Map Structure

GraphNav maps consist of waypoints, and edges between the waypoints. A waypoint consists of a reference frame, a name, a unique ID, and associated raw data. The raw data for a waypoint is stored in what is called a "Snapshot". Multiple waypoints may share the same snapshot.

Raw data includes feature clouds, April tag detections, imagery, terrain maps, etc.

Edges consist of a directed edge from one waypoint to another and a transform that estimates the relationship in 3D space between the two waypoints.

Maps do not have a global coordinate system (like GPS coordinates, for example). Only the relative transformations between waypoints are known.

## Understanding the Map Viewer

The map viewer displays waypoints as axes (red, green and blue arrows) where the "z" axis is blue, the "x" is red, and the "y" is green. They are connected by white lines representing edges.

Around the waypoints, the map viewer displays feature clouds. Feature clouds are collections of points that correspond to detected edge features in the robot's cameras. The feature clouds are colored by height, where blue is higher and red is lower.

The viewer also shows april tag detections as blue squares labeled with the fiducial ID. If multiple fiducials with the same ID are displayed near each other, this represents multiple detections taken at different times.
