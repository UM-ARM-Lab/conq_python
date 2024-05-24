# Boston Dynamics Sdk
import bosdyn.client
import bosdyn.client.lease
from bosdyn.client.image import ImageClient
from bosdyn.client.lease import LeaseClient
from bosdyn.client.robot_command import RobotCommandClient
from bosdyn.client.robot_command import RobotCommandBuilder

# Conq Manipulation
from conq.manipulation_lib.utils import verify_estop, stand

# Conq Memory
import conq.memory.src.memory_utils
from conq.memory.src.memory_utils import Memory

# Testing for the functions
if __name__ == "__main__":
    # Create an instance of the boston dynamics sdk with a name
    # I think these functions are included in everything that is in the bosdyn.client package
    sdk = bosdyn.client.create_standard_sdk('semantic_memory') 
    robot = sdk.create_robot('192.168.80.3')

    # This is the function that uses the username and password env vars to verify that we should have access to spot
    bosdyn.client.util.authenticate(robot)

    # This is a homebrewed function ensures a given robot is not currently e-stopped
    verify_estop(robot)

    image_client = robot.ensure_client(ImageClient.default_service_name)
    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)

    # I am fairly confident that this with statement's primary goal is to ensure that which ever device is currently leasing the robot is available through this entire function
    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        
        # Powering on sequence
        robot.logger.info('Powering on robot... This may take several seconds.')
        robot.power_on() # this function will ensure that the robot is powered on within the default time period of 20 seconds
        assert robot.is_powered_on(), "Robot failed to power on." # This is an assert statement that will ensure that the robot is properly powered on, otherwise it will raise an exception
        robot.logger.info('The robot has powered on.')

        # Create a client object which is just a "struct" of all of the other clients used on the robot packaged nicely
        
        robot.logger.info('Commanding the robot to stand')
        my_params = RobotCommandBuilder.mobility_params(body_height=0.5)
        stand(robot, command_client, params=my_params)

        # This should return a numpy array which contains the image from the 'right_fisheye_image' camera on spot
        memory = Memory(image_client)
        memory.observe_surroundings()
        memory.dream()
        memory._dump_json()