from typing import List, Tuple
from conq.navigation_lib.semantic_map_information.semantic_waypoint import SemanticWaypoint

class SemanticInformation:
    def __init__(self):
        self.semantic_waypoints: List[SemanticWaypoint] = []

    def create_waypoint(self, waypoint_name: str, coordinates: Tuple[float, float, float]):
        """
        Creates a new semantic waypoint with the given name and (x, y, z) coordinates.
        """
        # First check if the waypoint already by first seeing if the name is already in the list and if the coordinates are within 0.05 of each other
        for waypoint in self.semantic_waypoints:
            if waypoint.name == waypoint_name and abs(waypoint.coordinates[0] - coordinates[0]) < 0.05 and abs(waypoint.coordinates[1] - coordinates[1]) < 0.05 and abs(waypoint.coordinates[2] - coordinates[2] < 0.05):
                return
        self.semantic_waypoints.append(SemanticWaypoint(waypoint_name, coordinates))

    def prune_waypoints(self):
        # Add your implementation here
        pass