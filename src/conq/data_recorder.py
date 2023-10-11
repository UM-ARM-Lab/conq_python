import datetime
import json
import pickle
import time
from importlib.metadata import version
from multiprocessing import Event
from pathlib import Path
from threading import Thread

import numpy as np
from bosdyn.api import image_pb2
from bosdyn.api.robot_state_pb2 import FootState
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.client.robot_state import RobotStateClient

from conq.cameras_utils import RGB_SOURCES, DEPTH_SOURCES
from conq.rerun_utils import viz_common_frames


class LatestImageRecorder:
    """ Constantly queries the cameras and stores the latest image """

    def __init__(self, image_client: ImageClient, src, fmt, done: Event):
        self.image_client = image_client
        self.latest_img_path = None
        self.latest_img_res = None
        self.src = src
        self.fmt = fmt

        self.thread = Thread(target=self.save_imgs_worker, args=(image_client, src, fmt, done))

    def start(self):
        self.thread.start()

    def join(self):
        self.thread.join()

    def save_imgs_worker(self, image_client, src, fmt, done: Event):
        while not done.is_set():
            req = build_image_request(src, pixel_format=fmt)
            res = image_client.get_image([req])[0]

            self.latest_img_res = res


class ConqDataRecorder:

    def __init__(self, root: Path, robot_state_client: RobotStateClient, image_client: ImageClient):
        self.root = root
        self.robot_state_client = robot_state_client
        self.image_client = image_client

        self.root.mkdir(exist_ok=True, parents=True)
        self.metadata = {
            'bosdyn.api version': version("bosdyn.api"),
            'date':               datetime.datetime.now().isoformat(),
            'rgb_sources':        RGB_SOURCES,
            'depth_sources':      DEPTH_SOURCES,
        }
        self.metadata_path = self.root / 'metadata.json'
        with self.metadata_path.open('w') as f:
            json.dump(self.metadata, f, indent=2)

        self.imgs_done = Event()

        # These are the threads that constantly query the cameras and store the latest image.
        # They do not start/stop between episodes.
        self.img_recorders = [
            LatestImageRecorder(image_client, src, image_pb2.Image.PixelFormat.PIXEL_FORMAT_RGB_U8, self.imgs_done)
            for src in RGB_SOURCES
        ]
        self.img_recorders += [
            LatestImageRecorder(image_client, src, image_pb2.Image.PixelFormat.PIXEL_FORMAT_DEPTH_U16, self.imgs_done)
            for src in DEPTH_SOURCES
        ]

        for img_rec in self.img_recorders:
            img_rec.start()

        self.reset()

    def reset(self):
        self.robot_state_thread = None
        self.latest_instruction = None
        self.latest_instruction_time = None
        self.episode_idx = 0

    def start_episode(self, mode):
        self.episode_done = Event()
        self.robot_state_thread = Thread(target=self.save_episode_worker,
                                         args=(self.robot_state_client, self.episode_done, mode))
        self.robot_state_thread.start()

    def next_episode(self):
        self.episode_done.set()
        self.robot_state_thread.join()
        self.episode_idx += 1

    def stop(self):
        self.imgs_done.set()
        self.robot_state_thread.join()
        for img_rec in self.img_recorders:
            img_rec.join()

    def add_instruction(self, text: str):
        self.latest_instruction = text
        self.latest_instruction_time = time.time()

    def save_episode_worker(self, robot_state_client, done: Event, mode: str):
        mode_path = self.root / mode
        mode_path.mkdir(exist_ok=True, parents=True)

        episode_path = mode_path / f'episode_{self.episode_idx}.pkl'
        episode = []

        while not done.is_set():
            now = time.time()
            state = robot_state_client.get_robot_state()

            viz_common_frames(state.kinematic_state.transforms_snapshot)

            step_data = {
                'time':             now,
                'robot_state':      state,
                'instruction':      self.latest_instruction,
                'instruction_time': self.latest_instruction_time,
                'images':           {},
            }
            for rec in self.img_recorders:
                step_data['images'][rec.src] = rec.latest_img_res

            episode.append(step_data)
            print("saving...")

            # save and wait a bit so other threads can run
            with episode_path.open('wb') as f:
                pickle.dump(episode, f)


def get_state_vec(state):
    joint_positions = [js.position.value for js in state.kinematic_state.joint_states]
    joint_velocities = [js.velocity.value for js in state.kinematic_state.joint_states]
    body_vel = state.kinematic_state.velocity_of_body_in_vision
    body_vel_vec = [body_vel.linear.x, body_vel.linear.y, body_vel.linear.z,
                    body_vel.angular.x, body_vel.angular.y, body_vel.angular.z]
    is_holding_item = [float(state.manipulator_state.is_gripper_holding_item)]
    ee_force = state.manipulator_state.estimated_end_effector_force_in_hand
    ee_force_vec = [ee_force.x, ee_force.y, ee_force.z]

    def _fs_vec(fs: FootState):
        pos = fs.foot_position_rt_body
        return [pos.x, pos.y, pos.z, fs.contact]

    foot_states = np.reshape([_fs_vec(fs) for fs in state.foot_state], [-1])
    state_vec = np.concatenate([
        joint_positions,
        joint_velocities,
        body_vel_vec,
        is_holding_item,
        ee_force_vec,
        foot_states,
    ], dtype=np.float32)
    return state_vec
