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
from bosdyn.client.inverse_kinematics import InverseKinematicsClient

@dataclass
class Clients:
    lease: Optional[LeaseClient]
    state: Optional[RobotStateClient]
    manipulation: Optional[ManipulationApiClient]
    image: Optional[ImageClient]
    graphnav: Optional[GraphNavClient]
    raycast: Optional[RayCastClient]
    command: Optional[RobotCommandClient]
    robot: Optional[Robot]
    #ik: Optional[InverseKinematicsClient]