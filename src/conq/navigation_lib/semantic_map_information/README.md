# Semantic Map Information

Holds information about semantically important waypoints collected while operating. 

## SemanticWaypoint

Dataclass that holds the string name of the waypoints and SE3Pose of the waypoint.

## SemanticInformation

Class that contains and manages a list of `SemanticWaypoint` objects.

#### Functions

- `create_waypoint`: Takes in a human readable string representing the waypoint name and an SE3Pose. Adds a waypoint to the list `self.semantic_waypoints` if a waypoint with the same name and location does not exist. 
- `prune_waypoints`: Function is not implemented yet, but the intension is to have a function that will automatically prune waypoints that are likely representing the same semantic information when the list of waypoints gets large. 