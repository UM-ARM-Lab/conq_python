from click_map_interface import ClickMapInterface


# TODO: Make this a general class whose model could be OpenCV (as in find_plant_demo) or 
# a neural network (as in arm_segmentation)
class ObjectDetector:
    def __init__(self, model_path):
        # Load the model
        pass

    def detect(self, rgb_np):
        # Run the model on the image
        # Return a list of detections, each of which is a dict with keys "class", "mask", "confidence"
        pass


class ToolRetrievalInferface(ClickMapInterface):
    def __init__(self, robot, upload_path, silhouette=None, silhouetteActor=None):
        super().__init__(self, robot, upload_path, silhouette, silhouetteActor)
        self.detector = None

    def onKeyPressEvent(self, obj, event):
        key, actor =  super().onKeyPressEvent(obj, event)
        if key == 'g':
            # Go to a waypoint and pick up the tool, then come back
            if actor:
                initial_waypoint_id = self._get_localization_state()
                print(f"navigating to: {actor.waypoint_id}")
                self._navigate_to([actor.waypoint_id])
                self.find_tool() # may have to reorient the robot / run multiple times
                self._pick_up_tool()
                self._navigate_to([initial_waypoint_id])
            else: 
                print("No waypoint selected")

    def find_tool(self):
        # # get images from each camera
        # images = self._get_images()
        # # run the detector on each image
        # for image in images: 
        #     self.detector.detect(image)
        # # get the pixelwise location of the tool
        # # return pixel_coords
        pass

    def _pick_up_tool(self, pixel_coords):
        # Go to pick up the tool with highest detection confidence
        # See the find_plant_demo for an (unorganized) example of how to do this
        pass

