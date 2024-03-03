# export BOSDYN_CLIENT_USERNAME=user && export BOSDYN_CLIENT_PASSWORD=4f9y0eftzh76

import numpy as np
import time
import threading
import cv2

from conq.manipulation_lib.Manipulation import move_to_unblocking, open_gripper, close_gripper, move_to_blocking, get_camera_intrinsics,get_gpe_in_cam
from conq.manipulation_lib.utils import build_arm_target_from_vision

class VisualServoingController(threading.Thread):
    def __init__(self, gains, control_rate, pose_acquirer, clients):
        super().__init__()
        self.Kp, self.Kd = gains
        self.target_position = tuple(np.zeros(7))
        self.control_rate = control_rate
        self.pose_acquirer = pose_acquirer
        self.control_signal = np.zeros(6)
        self.command_pose = tuple(np.zeros(7))
        self.running = True
        self.lock = threading.Lock()

        self.last_time = None
        self.last_position = None
        self.last_error = None
        self.clients = clients

        # TODO: Get robot's current pose
        self.rob_current_pose = None

    def get_control_signal(self):
        with self.lock:
            return np.copy(self.control_signal),np.copy(self.command_pose)

    def stop(self):
        self.running = False

    def run(self):
        
        #start_time = time.time()
        visual_pose, delta_time = self.pose_acquirer.get_latest_pose() # Can be 0
        #current_time = time.time()
        arm_command_pose = build_arm_target_from_vision(self.clients,visual_pose) # Object pose in body frame

        # self.target_position = np.array(arm_command_pose)

        # pose_error = self.target_position - self.rob_current_pose

        # epsilon = 1e-5 
        # if self.last_position is not None and self.last_time is not None and abs(delta_time) > epsilon:
            
        #     delta_error = pose_error - (self.last_error if self.last_error is not None else np.array([0,0,0,0,0,0]))
        #     derivative = delta_error / delta_time
        # else:
        #     # delta_time # For first control iteration - Already handled in VisualPoseAcquirer Class
        #     derivative = np.array([0,0,0,0,0,0])

        # # PD Control
        # control_signal = self.Kp * pose_error  + self.Kd * derivative 
        # #print("Control Signal (Velocity m/s): ", control_signal)

        # # Update for next iteration
        # self.last_position = visual_pose
        # self.last_time = current_time
        # self.last_error = pose_error

        # delta_pose = control_signal * delta_time
        # # FIXME: Control signal needs to be added to the current_robot_pose!
        # command_pose = visual_pose + delta_pose
        # #print("Arm Command Pose: ", command_pose)

        with self.lock:
            #self.control_signal = self.control_signal
            self.command_pose = arm_command_pose

        # elapsed_time = time.time() - start_time
        # sleep_time = max(0, (1 / self.control_rate) - elapsed_time)
        # time.sleep(sleep_time)
