import cv2
import threading
import time
import numpy as np

from conq.cameras_utils import get_color_img
import apriltag

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

    def stop(self):
        self.running = False

    def get_latest_pose(self):
        with self.lock:
            return np.copy(self.visual_pose), np.copy(self.delta_time)

    def run(self):
        cv2.namedWindow("RGB", cv2.WINDOW_NORMAL)
        # Example configuration
        video_filename = 'output_video.avi'  # Specify the output video filename
        fps = 20.0  # Frames per second
        frame_size = (640, 480)  # Frame size as (width, height)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Specify the codec

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
                    self.rob_current_pose = None

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
    
    #TODO: Method for point cloud
    

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
    
    #print("Object center: \n", detection.center)
    #print("Object in camera frame: \n",object_pose[:-1,-1])

    # TODO / FIXME: Use transformation instead of shortcut for pose
    # Transformation from Camera Pose to Hand Pose
    object_in_hand = np.copy(object_pose)
    object_in_hand[0,-1]  = object_pose[2,-1] # X = Z
    object_in_hand[1,-1]  = -object_pose[0,-1] # Y = -X
    object_in_hand[2,-1]  = -object_pose[1,-1] # Z = -Y
    
    #print("Object in hand frame: \n",object_in_hand[:-1,-1])
    x_hand,y_hand,z_hand = object_in_hand[0,-1],object_in_hand[1,-1],object_in_hand[2,-1]
    roll_hand,pitch_hand,yaw_hand = 0,0,0
    visual_pose = (x_hand,y_hand,z_hand,roll_hand,pitch_hand,yaw_hand)

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