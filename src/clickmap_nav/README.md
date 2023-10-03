# Clickmap_nav: 
This is a combination of the view_map and graph_nav examples. You can use it to view a map that spot has made, and then click on the map to send commands.

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

2. Run the map viewer

```
python3 -m clickmap_nav <path_to_your_map_directory>
```
For example: 
```
python3 -m clickmap_nav ~/spot/maps/collabspace1/
```
## Camera Controls

- R : reset the camera
- Left Mouse: rotate the camera
- Right Mouse: zoom in/out.
- Middle Mouse: pan the camera.

## GraphNav Map Structure

GraphNav maps consist of waypoints, and edges between the waypoints. A waypoint consists of a reference frame, a name, a unique ID, and associated raw data. The raw data for a waypoint is stored in what is called a "Snapshot". Multiple waypoints may share the same snapshot.

Raw data includes feature clouds, April tag detections, imagery, terrain maps, etc.

Edges consist of a directed edge from one waypoint to another and a transform that estimates the relationship in 3D space between the two waypoints.

Maps do not have a global coordinate system (like GPS coordinates, for example). Only the relative transformations between waypoints are known.

## Understanding the Map Viewer

The map viewer displays waypoints as axes (red, green and blue arrows) where the "z" axis is blue, the "x" is red, and the "y" is green. They are connected by white lines representing edges.

Around the waypoints, the map viewer displays feature clouds. Feature clouds are collections of points that correspond to detected edge features in the robot's cameras. The feature clouds are colored by height, where blue is higher and red is lower.

The viewer also shows april tag detections as blue squares labeled with the fiducial ID. If multiple fiducials with the same ID are displayed near each other, this represents multiple detections taken at different times.
