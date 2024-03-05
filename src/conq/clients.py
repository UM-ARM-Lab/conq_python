from dataclasses import dataclass
from typing import Optional

from bosdyn.client.image import ImageClient
from bosdyn.client.lease import LeaseClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.ray_cast import RayCastClient
from bosdyn.client.robot import Robot
from bosdyn.client.robot_command import RobotCommandClient
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.graph_nav import GraphNavClient


@dataclass
class Clients:
    lease: Optional[LeaseClient]=None
    state: Optional[RobotStateClient]=None
    manipulation: Optional[ManipulationApiClient]=None
    image: Optional[ImageClient]=None
    graphnav: Optional[GraphNavClient]=None
    raycast: Optional[RayCastClient]=None
    command: Optional[RobotCommandClient]=None
    robot: Optional[Robot]=None
