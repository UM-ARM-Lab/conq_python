import datetime
import json
import pickle
import shutil
import time
from importlib.metadata import PackageNotFoundError, version
from multiprocessing import Event
from pathlib import Path
from threading import Thread
from typing import Callable, Optional

import numpy as np
from bosdyn.api.robot_state_pb2 import FootState
from bosdyn.client.image import build_image_request

from conq.cameras_utils import ALL_SOURCES, DEPTH_SOURCES, RGB_SOURCES, source_to_fmt
from conq.clients import Clients
from conq.rerun_utils import viz_common_frames


class ConqDataRecorder:

    def __init__(self, root: Path, clients: Clients, sources=None,
                 get_latest_action: Optional[Callable] = None, period: Optional[float] = None, map_directory_path: Optional[str] = None):
        """

        Args:
            root:
            robot_state_client:
            image_client:
            sources:
            get_latest_action: callable that returns the action we're currently commanding. For recording teleop demos.
        """
        self.period = period
        self.root = root
        self.clients = clients
        self.get_latest_action = get_latest_action

        if sources is None:
            self.sources = ALL_SOURCES
        else:
            self.sources = sources
        self.fmts = [source_to_fmt(src) for src in self.sources]

        self.root.mkdir(exist_ok=True, parents=True)
        try:
            api_version = version("bosdyn.api")
        except PackageNotFoundError:
            api_version = "unknown"
        self.metadata = {
            'bosdyn.api version': api_version,
            'date': datetime.datetime.now().isoformat(),
            'rgb_sources': RGB_SOURCES,
            'depth_sources': DEPTH_SOURCES,
            'period': self.period,
        }

        if map_directory_path is not None:
            map_path = Path(map_directory_path)
            self.copy_map_files(map_path)

        self.metadata_path = self.root / 'metadata.json'
        with self.metadata_path.open('w') as f:
            json.dump(self.metadata, f, indent=2)

        self.reset()

    def reset(self):
        self.saver_thread = None
        self.latest_instruction = "no instruction"
        self.latest_instruction_time = time.time()
        self.episode_idx = 0

    def start_episode(self, mode, instruction, save_interval: int = 50):
        self.episode_done = Event()
        self.saver_thread = Thread(target=self.save_episode_worker,
                                   args=(self.clients, self.episode_done, mode, save_interval))
        self.saver_thread.start()
        self.add_instruction(instruction)

    def next_episode(self):
        self.episode_done.set()
        self.saver_thread.join()
        self.episode_idx += 1

    def stop(self):
        self.episode_done.set()
        self.saver_thread.join()

    def add_instruction(self, text: str):
        self.latest_instruction = text
        self.latest_instruction_time = time.time()

    def save_episode_worker(self, clients, done: Event, mode: str, episode_save_interval):
        mode_path = self.root / mode
        mode_path.mkdir(exist_ok=True, parents=True)

        episode_path = mode_path / f'episode_{self.episode_idx}.pkl'
        episode = []

        while not done.is_set():
            now = time.time()
            state = None
            localization_state = None

            # Initialize step_data
            step_data = {
                'time': now,
                'instruction': self.latest_instruction,
                'instruction_time': self.latest_instruction_time,
                'images': {},
            }

            # Check that the right cleint being requested is set before setting the step_data
            if clients.state is not None:
                state = clients.state.get_robot_state()
                viz_common_frames(state.kinematic_state.transforms_snapshot)
                step_data['robot_state'] = state

            if clients.graphnav is not None:
                localization_state = clients.graphnav.get_localization_state()
                step_data['localization'] = localization_state.localization
                step_data['is_lost'] = localization_state.lost_detector_state.is_lost

            if clients.image is not None:
                reqs = []
                for src, fmt in zip(self.sources, self.fmts):
                    req = build_image_request(src, pixel_format=fmt)
                    reqs.append(req)

                ress = clients.image.get_image(reqs)

                for res, src in zip(ress, self.sources):
                    step_data['images'][src] = res

            # Get the action at the end, so we can see what the demonstrator was trying to do most recently.
            if self.get_latest_action is not None:
                step_data['action'] = self.get_latest_action(now)

            episode.append(step_data)

            if self.period is not None:
                # sleep to achieve the desired frequency
                sleep_dt = self.period - (time.time() - now)
                if sleep_dt > 0:
                    time.sleep(sleep_dt)

            # save
            if len(episode) % episode_save_interval == 0:
                with episode_path.open('wb') as f:
                    pickle.dump(episode, f)

        if len(episode) < 15:
            print(f"WARNING: episode {self.episode_idx} has only {len(episode)} steps!")
            print("This may mean that the demonstrator mis-clicked and did not actually perform the task.")

        # ensure we save before exiting or moving on to the next episode
        with episode_path.open('wb') as f:
            pickle.dump(episode, f)
    
    def copy_map_files(self, map_path: Path):
        if map_path.exists():
            print(f"MAP PATH: {map_path}")
            dst = self.root / map_path.name
            if map_path.is_dir():
                shutil.copytree(map_path, dst)
            else:
                shutil.copyfile(map_path, dst)
            self.metadata['map_path'] = dst.as_posix()
        else:
            print(f"WARNING from data_recorder.py: map path {map_path} does not exist! Not adding path to metadata.")



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