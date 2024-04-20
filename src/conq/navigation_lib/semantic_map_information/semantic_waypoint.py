from dataclasses import dataclass
from typing import Tuple

@dataclass
class SemanticWaypoint:
    """
    Represents a semantic waypoint with a name and coordinates.
    """

    name: str
    coordinates: Tuple[float, float, float]