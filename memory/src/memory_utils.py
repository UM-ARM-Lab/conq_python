# Boston Dynamics Sdk
import bosdyn.client
import bosdyn.client.lease
from bosdyn.client.image import ImageClient
from bosdyn.client.lease import LeaseClient
from bosdyn.client.robot_command import RobotCommandClient

# Conq perception
from collections import namedtuple
from conq.cameras_utils import get_color_img

# Conq Manipulation
from conq.manipulation_lib.utils import verify_estop, stand

# Misc
import cv2
import numpy as np
import json

# Constant values for pathing
MEMORY_IMAGE_PATH = './memory/memory_images/'
MEMORY_JSON_PATH = './memory/json/memory.json'

# Constant for all of the sources spot will be using to gather information on its surroundings
SOURCES = ['right_fisheye_image', 'left_fisheye_image', 'frontright_fisheye_image', 'frontleft_fisheye_image', 'back_fisheye_image']

Item = namedtuple('Item', ['x', 'y'])

class Memory:
    def __init__(self, image_client):
        # Save the image client for spot
        self.image_client = image_client

        # Load the existing memory json file into a dictionary of Item tuples
        self._load_json()

    # The _ at the beginning of this member function is supposed to tell people that it is a "private" member function to the Memory class
    def _store_lens(self, source):
        # Use the existing conq code to grab images in order to read images from conq's cameras
        image, _ = get_color_img(self.image_client, source)
        image = np.array(image,dtype=np.uint8)

        # Printing out the shape of the image
        dims = np.shape(image)
        print(f"Saving an image of size: {dims[0]} {dims[1]} {dims[2]} from source {source}")
    
        # Save the image
        cv2.imwrite(MEMORY_IMAGE_PATH + source + ".jpg", image)

    # This function will be used for writing to json files
    def _dump_json(self):
        json_dump = json.dumps(self.object_dict, indent = 4)
        with open(MEMORY_JSON_PATH, 'w') as json_file:
            json_file.write(json_dump)
    # This function writes all of the objects that spot has taken note of to a json file
    def _load_json(self):
        with open(MEMORY_JSON_PATH, 'r') as json_file:
            dictOfLists = json.load(json_file)
            self.object_dict = {key: Item(dictOfLists[key][0], dictOfLists[key][1]) for key in dictOfLists}

    # This function calls _storeLens on all of the sources to capture the images in each of the lens
    def observe_surroundings(self):
        # Save all of the images from conq's sources in the temporary image folder
        for source in SOURCES:
            self._store_lens(source)

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
        stand(robot, command_client)

        # This should return a numpy array which contains the image from the 'right_fisheye_image' camera on spot
        memory = Memory(image_client)
        memory._dump_json()
        memory._load_json()
        memory._dump_json()
        memory.observe_surroundings()

