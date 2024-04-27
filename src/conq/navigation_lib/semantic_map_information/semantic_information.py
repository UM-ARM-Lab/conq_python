from typing import List, Tuple

from bosdyn.client.math_helpers import SE3Pose

from conq.navigation_lib.semantic_map_information.semantic_waypoint import (
    SemanticWaypoint,
)


class SemanticInformation:
    def __init__(self):
        self.semantic_waypoints: List[SemanticWaypoint] = []

    def create_waypoint(self, waypoint_name: str, coordinates: SE3Pose):
        """
        Creates a new semantic waypoint with the given name and (x, y, z) coordinates.
        """
        # First check if the waypoint already by first seeing if the name is already in the list and if the coordinates are within 0.05 of each other
        for waypoint in self.semantic_waypoints:
            # TODO: Will need to refine this because chat gpt might give a different response based on what the image captures
            if waypoint.name == waypoint_name or (
                abs(waypoint.coordinates.position.x - coordinates.position.x) < 0.05
                and abs(waypoint.coordinates.position.y - coordinates.position.y) < 0.05
            ):
                print(
                    "create_waypoint: Already have a waypoint with the same name or coordinates"
                )
                return
        self.semantic_waypoints.append(SemanticWaypoint(waypoint_name, coordinates))
        print(f"Added semantic information to the map: {self.semantic_waypoints[-1]}")

    def prune_waypoints(self):
        # TODO: Add a way to prune waypoints that are too close to each other
        pass
