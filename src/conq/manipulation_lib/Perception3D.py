import cv2
import threading
import time
import numpy as np
import open3d as o3d
from typing import Union, Tuple, Optional

# CONQ: Perception-> Object Track/Pose Estimator
import apriltag # FIXME: Temporary until integrated with 6-DOF object pose estimator 

# BOSDYN
from bosdyn.client.math_helpers import Quat, quat_to_eulerZYX
from bosdyn.client.image import ImageClient, depth_image_to_pointcloud, _depth_image_data_to_numpy
from conq.cameras_utils import get_color_img, get_depth_img, pos_in_cam_to_pos_in_hand, image_to_opencv, RGB_SOURCES, DEPTH_SOURCES, rotate_image
from bosdyn.client.frame_helpers import VISION_FRAME_NAME, get_a_tform_b
from bosdyn.api.image_pb2 import ImageResponse, _IMAGERESPONSE
from conq.manipulation_lib.utils import depth2pcl, pcl_transform

class VisualPoseAcquirer(threading.Thread):
    def __init__(self, image_client, sources, camera_params):
        super().__init__()
        self.image_client = image_client
        self.sources = sources
        self.camera_params = camera_params
        self.running = True
        self.lock = threading.Lock()
        self.visual_pose = tuple(np.zeros(6))
        self.delta_time = 0.0
        self.latest_image = np.zeros(shape=(480,640,3))
        self.latest_depth_response = []
        self.latest_response = []
        # self.rgb_updated_event = threading.Event()  # Event to signal when RGB image is updated
        
    def stop(self):
        self.running = False

    def run(self):
        cv2.namedWindow("RGB", cv2.WINDOW_NORMAL)
        # Example configuration
        video_filename = 'output_video.avi'  # Specify the output video filename
        fps = 20.0  # Frames per second
        frame_size = (640, 480)  # Frame size as (width, height)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # codec

        # Initialize the VideoWriter object
        out = cv2.VideoWriter(video_filename, fourcc, fps, frame_size)

        while self.running:
            try:
                image_responses = self.image_client.get_image_from_sources(self.sources)
                start_time = time.time()
                img_bgr, rgb_response = get_Image(self.image_client, self.sources[1])
                result, _, img_bgr, visual_pose, object_center = get_object_pose(img_bgr, rgb_response, self.camera_params)
                current_time = time.time()
                
                delta_time = current_time - start_time

                with self.lock:
                    self.visual_pose = visual_pose
                    self.delta_time = delta_time
                    self.latest_image = img_bgr
                    # self.rgb_updated_event.set()
                    self.latest_response = image_responses
                    
            except IndexError as i_e:
                #print("Target object not in frame")
                current_time = time.time()
                delta_time = current_time - start_time
                with self.lock:

                    self.delta_time = delta_time
                pass
            except Exception as e:
                if self.visual_pose is None:
                    pass
                else:
                    print(e)
                    break
            
            cv2.imshow("RGB", img_bgr)
            # Write the frame to the video file
            # out.write(img_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Process events
                break   
            #time.sleep(0.01)  # Adjust as needed based on your acquisition rate
        # out.release()
        cv2.destroyAllWindows()

    def get_latest_pose(self):
        with self.lock:
            return np.copy(self.visual_pose), np.copy(self.delta_time)
        
    def get_latest_RGB(self,path, save=False, file_name = "live"):
        # Wait until the RGB image is updated
        # self.rgb_updated_event.wait()
        
        #print("Latest response: ", self.get_latest_RGB_response())
        with self.lock:
            rgb_image = np.copy(self.latest_image)
            print("Latest image: ", self.latest_image)
            print("Acquisition time: ", self.delta_time)
            if save:
                extension = file_name+".jpg"
                #cv_visual = cv2.imdecode(np.frombuffer(self.get_latest_RGB_response().shot.image.data, dtype=np.uint8), -1)
                cv2.imwrite(path + extension, rgb_image)
                print(f"File written in {path + extension}")
            # Reset the event for the next update
            # self.rgb_updated_event.clear()
            return rgb_image
    
    def get_latest_RGB_response(self):
        with self.lock:
            return self.latest_response[1]
    
    def get_latest_depth_response(self):
        image_responses = self.image_client.get_image_from_sources(self.sources)
        if image_responses is not None:
            # Ensure there's a depth response to return
            print("Depth response: ", type(image_responses[0]))
            return image_responses[0]
        else:
            # Log or handle the absence of a depth response appropriately
            print("Depth response is not available.")
            return None

class Vision:
    def __init__(self, image_client, sources):
        self.image_client = image_client
        self.sources = sources
        self.running = True
        self.visual_pose = tuple(np.zeros(6))
        self.delta_time = 0.0
        self.latest_image = np.zeros(shape=(480,640,3))
        self.latest_depth_response = []
        self.latest_response = []

    def get_latest_response(self):
        self.latest_response = self.image_client.get_image_from_sources(self.sources)
        return self.latest_response

    def get_latest_RGB(self,path=None, save=False, file_name = "live"):
        img_bgr, _ = get_Image(self.image_client, self.sources[1])
        self.latest_image = img_bgr
        if save:
            extension = file_name+".jpg"
            #cv_visual = cv2.imdecode(np.frombuffer(self.get_latest_RGB_response().shot.image.data, dtype=np.uint8), -1)
            cv2.imwrite(path + extension, img_bgr)
            print(f"File written in {path + extension}")

        return img_bgr
    
    def get_latest_Depth(self,path=None,save=False, file_name="live"):
        depth_frame = _depth_image_data_to_numpy(image_response=self.get_latest_response()[0]) # Need this for aligning RGB and Depth
        print("Shape of Depth frame: ", np.shape(depth_frame))
        if save:
            # Array
            extension = file_name+".npy"
            np.save(path + extension, depth_frame)
            print(f"File written in {path + extension}")
            # Image
            cv_depth = np.frombuffer(self.get_latest_response()[0].shot.image.data,dtype=np.uint16)
            cv_depth = cv_depth.reshape(self.get_latest_response()[0].shot.image.rows,
                                        self.get_latest_response()[0].shot.image.cols)
            
            min_val = np.min(cv_depth)
            max_val = np.max(cv_depth)
            depth_range = max_val - min_val
            depth8 = (255.0 / depth_range * (cv_depth - min_val)).astype('uint8')
            depth8_rgb = cv2.cvtColor(depth8, cv2.COLOR_GRAY2RGB)
            extension = file_name+".jpg"
            cv2.imwrite(path + extension, depth8_rgb)
            print(f"File written in {path + extension}")

        return depth_frame
    
    def segment_image(self,depth_image, seg_mask):
        """Segment the depth image (aligned with RGB frame) based on the segmentation mask."""
        segmented_depth_image = depth_image.copy()
        segmented_depth_image[~seg_mask] = 0  # Set non-segmented regions to zero
        return segmented_depth_image
    
class PointCloud:
    def __init__(self, vision):
        self.xyz = None
        self.vision = vision
        self.sources = vision.sources

    def get_raw_point_cloud(self,min_dist=0,max_dist=10, target_frame = "body"):
        
        image_response = self.vision.get_latest_response()[0]
        if image_response is None:
            # raise ImageResponseUnavailableError("No image response available for point cloud generation.")
            print("get_point_cloud: No image response available for point cloud generation.")
            return self.xyz
        else:
            
            sensor_pcl = depth_image_to_pointcloud(image_response=image_response, min_dist = min_dist, max_dist = max_dist)
            # Transform pcl to target_frame
            target_pcl_sensor = pcl_transform(sensor_pcl, image_response, source = self.sources[0], target_frame = target_frame)
            self.xyz = target_pcl_sensor
            return target_pcl_sensor
        
    def process(self,seg_mask,target_frame):
        """Process raw pointcloud: 1) Segment pointcloud from Lang-SAM"""
        
        return self.xyz
    
    def segment_xyz(self, seg_mask, target_frame = "body", min_dist=0,max_dist=10):
        """Segment depth image and pointcloud"""
        # Get depth image (RGB-aligned)
        depth_image = self.vision.get_latest_Depth()
        # Depth response:
        depth_response = self.vision.get_latest_response()[0]
        # DEBUGGING: Segmented depth image
        segmented_depth = self.vision.segment_image(depth_image, seg_mask)

        sensor_pcl = depth2pcl(depth_response, seg_mask, min_dist, max_dist)
        # Transform pcl to target_frame
        target_pcl_sensor = pcl_transform(sensor_pcl, depth_response, source = self.sources[0], target_frame = target_frame)
        
        self.xyz = target_pcl_sensor
        return target_pcl_sensor
    
    def get_pcd(self):
        pcd = None
        if self.xyz is None:
            # raise PointCloudDataUnavailableError("Point cloud data not available.")
            print("No image response available for point cloud generation.")
            pass
        else:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.xyz)
            return pcd
        
    def save_pcd(self, path, file_name="live"):
        """Saves the current point cloud data to a PCD file."""
        start_time = time.time()
        if self.xyz is None:
            print("No point cloud data!")
        pcd_file_path = f"{path}{file_name}.pcd"
        with open(pcd_file_path, 'w') as pcd_file:
            # Write the PCD header
            pcd_file.write("VERSION .7\n")
            pcd_file.write("FIELDS x y z\n")
            pcd_file.write(f"SIZE 4 4 4\n")
            pcd_file.write(f"TYPE F F F\n")
            pcd_file.write(f"COUNT 1 1 1\n")
            pcd_file.write(f"WIDTH {len(self.xyz)}\n")
            pcd_file.write("HEIGHT 1\n")
            pcd_file.write("VIEWPOINT 0 0 0 1 0 0 0\n")
            pcd_file.write(f"POINTS {len(self.xyz)}\n")
            pcd_file.write("DATA ascii\n")
            # Write the point data
            for point in self.xyz:
                pcd_file.write(f"{point[0]} {point[1]} {point[2]}\n")
            end_time = time.time()
            print(f"File written in {pcd_file_path} in {round((end_time - start_time),2)} s")

    def save_npy(self,path, file_name="live"):
        """Saves the pointcloud as a Nx3 numpy array"""
        start_time = time.time()
        np.save(f'{path}{file_name}.npy', self.xyz)
        end_time = time.time()
        print(f"File written in {path}{file_name}.npy in {round((end_time - start_time),2)} s")

    def run(self):
        # super().run() # Required if need continuous point cloud acquisition
        pass

class ImageResponseUnavailableError(Exception):
    """Exception raised when the image response is not available for point cloud generation."""
    pass

class PointCloudDataUnavailableError(Exception):
    """Exception raised when point cloud data is not available or not yet generated."""
    pass


# TODO: Move to utils
def get_Image(image_client, sources):
    rgb_np, rgb_response = get_color_img(image_client, sources)
    rgb_np = np.array(rgb_np, dtype=np.uint8)

    img_bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR) 
    return img_bgr, rgb_response
    
def get_object_pose(img_bgr, rgb_response, camera_params):
            
    result, overlay = apriltag_image(img_bgr, camera_params, tag_size=0.0635) # Replace with object_pose estimator

    ### ______________________###
    object_pose = result[1]
                
    detection = result[0]
    
    #print("Object pose: \n", object_pose)
    #print("Object in camera frame: \n",object_pose[:-1,-1])

    # TODO / FIXME: Use transformation instead of shortcut for pose
    # Transformation from Camera Pose to Hand Pose
    object_in_hand = np.copy(object_pose)
    hand_T_camera = np.array([[0,0,1,0],
                              [-1,0,0,0],
                              [0,-1,0,0],
                              [0,0,0,1]])
    
    object_in_hand = np.matmul(hand_T_camera,object_pose)
    print("Object in hand: \n", object_in_hand)

    quat = Quat.from_matrix(object_in_hand[:-1,:-1])
    print("Rotation in quaternion: ", quat)

    yaw_hand, pitch_hand,roll_hand = quat_to_eulerZYX(quat)

    # object_in_hand[0,-1]  = object_pose[2,-1] # X_hand = Z_camera
    # object_in_hand[1,-1]  = -object_pose[0,-1] # Y_hand = -X_camera
    # object_in_hand[2,-1]  = -object_pose[1,-1] # Z_hand = -Y_camera
    
    #print("Object in hand frame: \n",object_in_hand[:-1,-1])
    x_hand,y_hand,z_hand = object_in_hand[0,-1],object_in_hand[1,-1],object_in_hand[2,-1]
    #roll_hand,pitch_hand,yaw_hand = 0,0,0
    visual_pose = (x_hand,y_hand,z_hand,roll_hand,pitch_hand,yaw_hand)
    #visual_pose = (x_hand,y_hand,z_hand,quat.w,quat.x,quat.y,quat.z)

    ## Overlays
    object_center = (int(detection.center[0]),int(detection.center[1]))   # Assuming this gives the center as [x, y]

    # Draw circular marker at object center
    cv2.circle(overlay, object_center, 5, (0, 255, 0), -1)

    # Image center
    image_center = [int(overlay.shape[1] // 2), int(overlay.shape[0] // 2)]
    cv2.circle(overlay, tuple(image_center), 5, (255, 0, 0), -1)

    # Draw arrow from image center to object center
    cv2.arrowedLine(overlay, tuple(image_center), tuple(object_center), (0, 0, 255), 2)

    return result, rgb_response, overlay, visual_pose, object_center

def apriltag_image(img, camera_params, tag_size):

    '''
    Detect AprilTags from static images.

    Args:   input_images: cv image
            camera_params [list[float]]: [fx,fy,cx,cy]
    '''


    '''
    Set up a reasonable search path for the apriltag DLL.
    Either install the DLL in the appropriate system-wide
    location, or specify your own search paths as needed.
    '''

    detector = apriltag.Detector(options=None, searchpath=apriltag._get_dll_path())

    result, overlay = apriltag.detect_tags(img,
                                            detector,
                                            camera_params,
                                            tag_size,
                                            vizualization=3,
                                            verbose=None,
                                            annotation=True
                                            )
    return result, overlay