# Conq perception
from conq.cameras_utils import get_color_img

# Misc
import cv2
import numpy as np
import json

# Constant values for pathing
MEMORY_IMAGE_PATH = './memory/memory_images/'
MEMORY_JSON_PATH = './memory/json/memory.json'

# Constant for all of the sources spot will be using to gather information on its surroundings
SOURCES = ['right_fisheye_image', 'left_fisheye_image', 'frontright_fisheye_image', 'frontleft_fisheye_image', 'back_fisheye_image']

class Waypoint:
    def __init__(self, index):
        self.index = index
        self.objects = []
        self.description = "no description provided"

class Memory:
    def __init__(self, image_client):
        self.image_client = image_client
        # TODO: Make sure that this constructor properly reads the json file
        self.json_list = ["words", "more", "words"]

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
        json_dump = json.dumps(self.json_list, indent = 4)
        with open(MEMORY_JSON_PATH) as json_file:
            json_file.write(json_dump)

    # This function calls _storeLens on all of the sources to capture the images in each of the lens
    def observe_surroundings(self):
        # Save all of the images from conq's sources in the temporary image folder
        for source in SOURCES:
            self._store_lens(source)
    
    def write_object_to_waypoint(self, waypoint_index):
        assert waypoint_index < len(self.json_list), "waypoint_index is out of bounds for json_list"
        waypoint = self.json_list[waypoint_index]

    def add_waypoint(self):
        new_waypoint = Waypoint(len(self.json_list))
        self.json_list.append(new_waypoint) # I think pyright is buggin here

memory = Memory("image_client")
memory._dump_json()
