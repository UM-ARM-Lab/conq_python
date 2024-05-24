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

# Conq Memory
from conq.memory.src.dream_utils import Dreaming

# Misc
import cv2
import numpy as np
import json

# Constant values for pathing
MEMORY_IMAGE_PATH = './data/memory_images/'
MEMORY_JSON_PATH = './data/json/memory.json'
AREA_MEMORY_JSON_PATH = './data/json/area_memory.json'

# Constant for all of the sources spot will be using to gather information on its surroundings
SOURCES = ['right_fisheye_image', 'left_fisheye_image', 'back_fisheye_image']

EXAMPLE_JSON = """
{
    "chair": [
        "chair",
        [
            "lounge",
            "seating area"
        ],
        [
            1,
            2,
            3
        ]
    ]
}
"""

NAME_INDEX = 0
AREA_INDEX = 1
WAYPOINT_INDEX = 2

Item = namedtuple('Item', ['name', 'area_dict', 'waypoints_dict'])

class Memory:
    def __init__(self, image_client):
        # Create the dreaming object that memory will interface with
        self.dreamer = Dreaming()
        
        # Save the image client for spot
        self.image_client = image_client

        # This is for testing purposes only
        self._clear_object_json()
        self._clear_area_json()

        # Load existing memory on objects and areas into memory
        self._load_object_json()
        self._load_area_json()

    #### PRIVATE MEMBER FUNCTIONS ####

    # The _ at the beginning of this member function is supposed to tell people that it is a "private" member function to the Memory class
    def _store_lens(self, source, waypoint_id):
        # Use the existing conq code to grab images in order to read images from conq's cameras
        image, _ = get_color_img(self.image_client, source)
        image = np.array(image,dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Printing out the shape of the image
        dims = np.shape(image)
        print(f"Saving an image of size: {dims[0]} {dims[1]} {dims[2]} from source {source}")
    
        # Save the image
        cv2.imwrite(MEMORY_IMAGE_PATH + source + '-' + str(waypoint_id) + ".jpg", image)

    # This function will be used for writing to json files
    def _dump_object_json(self):
        self.object_dict = {key: Item(self.object_dict[key][0], list(self.object_dict[key][1]), list(self.object_dict[key][2])) for key in self.object_dict}
        json_dump = json.dumps(self.object_dict, indent = 4)
        with open(MEMORY_JSON_PATH, 'w') as json_file:
            json_file.write(json_dump)

    # This function writes all of the objects that spot has taken note of to a json file
    def _load_object_json(self):
        with open(MEMORY_JSON_PATH, 'r') as json_file:
            dictOfLists = json.load(json_file)
            self.object_dict = {key: Item(dictOfLists[key][0], set(dictOfLists[key][1]), set(dictOfLists[key][2])) for key in dictOfLists}

    # This function writes a empty json object to the object json file
    def _clear_object_json(self):
        # Write an empty JSON object to the file
        with open(MEMORY_JSON_PATH, 'w') as json_file:
            json.dump({}, json_file, indent=4)

    # This function puts all of the semantic areas that are stored inside of the json file into a dictionary
    def _load_area_json(self):
        with open(AREA_MEMORY_JSON_PATH, 'r') as json_file:
            self.area_dict = json.load(json_file)

    # This function writes a empty json object to the area json file
    def _clear_area_json(self):
        # Write an empty JSON object to the file
        with open(AREA_MEMORY_JSON_PATH, 'w') as json_file:
            json.dump({}, json_file, indent=4)

    # This function will be used for writing to json files
    def _dump_area_json(self):
        json_dump = json.dumps(self.area_dict, indent = 4)
        with open(AREA_MEMORY_JSON_PATH, 'w') as json_file:
            json_file.write(json_dump)

    # Either creates a new object in the dictionary or adds the corresponding waypoints to the set
    def _add_object(self, item):
        if item.name in self.object_dict.keys():
            # Append the observed waypoint to the master waypoints
            for waypoint_id in item.waypoints_dict:
                self.object_dict[item.name][WAYPOINT_INDEX].add(waypoint_id)
        else:
            self.object_dict[item.name] = item

    # Updates all of the areas inside of the set to correspond to the appropriate waypoint
    def _add_area(self, item):
        for area in item.area_dict:
            self.area_dict[area] = item.waypoints_dict[0]

    #### PUBLIC MEMBER FUNCTIONS ####

    # This function calls _storeLens on all of the sources to capture the images in each of the lens
    def observe_surroundings(self):
        # Save all of the images from conq's sources in the temporary image folder
        for source in SOURCES:
            self._store_lens(source, 1)
            
    # This function begins the dreaming sequence
    def dream(self):
        while self.dreamer.can_dream():
            waypoint_id, curr_objects, curr_areas = self.dreamer.detect_objects()
            print(waypoint_id, curr_objects, curr_areas)
            
            for obj in curr_objects:
                if not obj == 'None':
                    new_item = Item(obj, curr_areas, [waypoint_id])
                    self._add_object(new_item)
                    self._add_area(new_item)
