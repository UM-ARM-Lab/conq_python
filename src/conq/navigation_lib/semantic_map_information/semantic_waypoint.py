from dataclasses import dataclass
from typing import Tuple

from bosdyn.client.math_helpers import SE3Pose


@dataclass
class SemanticWaypoint:
    """
    Represents a semantic waypoint with a name and coordinates in x,y,z.
    """

    name: str
    coordinates: SE3Pose
