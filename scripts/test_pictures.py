# Boston dynamics
# Estop Imports
import bosdyn.client.estop
from bosdyn.client.estop import EstopClient # This imports a specific class into the script rather
from bosdyn.client.estop import estop_pb2

# Clients
from bosdyn.client.image import ImageClient
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient,
                                         block_until_arm_arrives, blocking_stand)
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.ray_cast import RayCastClient
import bosdyn.client.lease
from bosdyn.client.lease import LeaseClient

# Conq
from conq.clients import Clients
from conq.manipulation_lib.utils import stand

def verify_estop(robot):
    """Verify the robot is not estopped"""

    client = robot.ensure_client(EstopClient.default_service_name)
    if client.get_status().stop_level != estop_pb2.ESTOP_LEVEL_NONE:
        error_message = "Robot is estopped. Please use an external E-Stop client, such as the" \
        " estop SDK example, to configure E-Stop."
        robot.logger.error(error_message)
        raise Exception(error_message)

if __name__ == "__main__":
    sources = 

    # Create an instance of the boston dynamics sdk with a name
    # I think these functions are included in everything that is in the bosdyn.client package
    sdk = bosdyn.client.create_standard_sdk('semantic_memory') 
    robot = sdk.create_robot('192.168.80.3')

    # This is the function that uses the username and password env vars to verify that we should have access to spot
    bosdyn.client.util.authenticate(robot)

    # This is a homebrewed function ensures a given robot is not currently e-stopped
    verify_estop(robot)

    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    manipulation_api_client = robot.ensure_client(ManipulationApiClient.default_service_name)
    image_client = robot.ensure_client(ImageClient.default_service_name)
    rc_client = robot.ensure_client(RayCastClient.default_service_name)
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)

    vision = Vision(image_client, sources)

    # I am fairly confident that this with statement's primary goal is to ensure that which ever device is currently leasing the robot is available through this entire function
    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        
        # Powering on sequence
        robot.logger.info('Powering on robot... This may take several seconds.')
        robot.power_on() # this function will ensure that the robot is powered on within the default time period of 20 seconds
        assert robot.is_powered_on(), "Robot failed to power on." # This is an assert statement that will ensure that the robot is properly powered on, otherwise it will raise an exception
        robot.logger.info('The robot has powered on.')

        # Create a client object which is just a "struct" of all of the other clients used on the robot packaged nicely
        clients = Clients(lease=lease_client, state=robot_state_client, manipulation=manipulation_api_client,
                            image=image_client, raycast=rc_client, command=command_client, robot=robot, graphnav=None)
        
        robot.logger.info('Commanding the robot to stand')
        stand(robot, command_client)

        