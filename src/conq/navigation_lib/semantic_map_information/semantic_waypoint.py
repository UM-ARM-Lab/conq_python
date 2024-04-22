from dataclasses import dataclass
from typing import Tuple

@dataclass
class SemanticWaypoint:
    """
    Represents a semantic waypoint with a name and coordinates in x,y,z.
    """

    name: str
    coordinates: Tuple[float, float, float]