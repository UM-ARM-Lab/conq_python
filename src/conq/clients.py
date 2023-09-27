from dataclasses import dataclass

from bosdyn.client.robot import Robot
from bosdyn.client.image import ImageClient
from bosdyn.client.lease import LeaseClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.ray_cast import RayCastClient
from bosdyn.client.robot_command import RobotCommandClient
from bosdyn.client.robot_state import RobotStateClient


@dataclass
class Clients:
    lease: LeaseClient
    state: RobotStateClient
    manipulation: ManipulationApiClient
    image: ImageClient
    raycast: RayCastClient
    command: RobotCommandClient
    robot: Robot
