import cv2
import threading
import time
import numpy as np
import open3d as o3d

# CONQ: Perception-> Object Track/Pose Estimator
import apriltag # FIXME: Temporary until integrated with 6-DOF object pose estimator 

# BOSDYN
from bosdyn.client.math_helpers import Quat, quat_to_eulerZYX
from bosdyn.client.image import ImageClient, depth_image_to_pointcloud, _depth_image_data_to_numpy
from conq.cameras_utils import get_color_img, get_depth_img, pos_in_cam_to_pos_in_hand, image_to_opencv, RGB_SOURCES, DEPTH_SOURCES
from bosdyn.client.frame_helpers import VISION_FRAME_NAME

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
        self.latest_image_response = None
        
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
                start_time = time.time()
                img_bgr, rgb_response = get_Image(self.image_client, self.sources[1])
                result, rgb_response, img_bgr, visual_pose, object_center = get_object_pose(img_bgr, rgb_response, self.camera_params)
                current_time = time.time()

                delta_time = current_time - start_time

                with self.lock:
                    self.visual_pose = visual_pose
                    self.delta_time = delta_time
                    self.latest_image = img_bgr
                    self.latest_image_response = rgb_response

            except IndexError as i_e:
                print("Target object not in frame")
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
            out.write(img_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Process events
                break   
            #time.sleep(0.01)  # Adjust as needed based on your acquisition rate
        out.release()
        cv2.destroyAllWindows()

    def get_latest_pose(self):
        with self.lock:
            return np.copy(self.visual_pose), np.copy(self.delta_time)
        
    def get_latest_image(self):
        with self.lock:
            return np.copy(self.latest_image)
    
    def get_latest_image_response(self):
        with self.lock:
            return np.copy(self.latest_image_response)
    
class PointCloud(VisualPoseAcquirer):
    def __init__(self, image_client, sources, camera_params):
        super().__init__(image_client, sources, camera_params)
        self.xyz = None

    def get_point_cloud(self,min_dist=0,max_dist=10):
        if self.latest_image_response is None:
            raise ImageResponseUnavailableError("No image response available for point cloud generation.")
        with self.lock:
            self.xyz = depth_image_to_pointcloud(image_response=self.get_latest_image_response()[0], min_dist = min_dist, max_dist = max_dist)
            # TODO: Perform transformations from sensor frame to HAND FRAME / BODY ALIGNED FRAME
            # TODO: Process pointcloud
            return self.xyz
    
    def get_pcd(self):
        if self.xyz is None:
            raise PointCloudDataUnavailableError("Point cloud data not available.")
        with self.lock:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.xyz)
            return pcd
        
    def process(self,xyz):
        """Process raw pointcloud """
        pass
        
    def save_pcd(self, pcd_file_path):
        """Saves the current point cloud data to a PCD file."""
        if self.xyz is None:
            raise PointCloudDataUnavailableError("Point cloud data not available for saving.")
        with self.lock:
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