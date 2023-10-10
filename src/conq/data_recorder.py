import datetime
import json
import pickle
import time
from importlib.metadata import version
from multiprocessing import Event
from pathlib import Path
from threading import Thread

from bosdyn.api import image_pb2
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

        self.thread = Thread(target=self.save_imgs, args=(image_client, src, fmt, done))

    def start(self):
        self.thread.start()

    def join(self):
        self.thread.join()

    def save_imgs(self, image_client, src, fmt, done: Event):
        while not done.is_set():
            req = build_image_request(src, pixel_format=fmt)
            res = image_client.get_image([req])[0]

            self.latest_img_res = res


class ConqDataRecorder:

    def __init__(self, root: Path, robot_state_client: RobotStateClient, image_client: ImageClient):
        self.root = root

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

        # event to kill threads
        self.done = Event()

        self.robot_state_thread = Thread(target=self.save_robot_state,
                                         args=(robot_state_client, self.done))
        self.img_recorders = [
            LatestImageRecorder(image_client, src, image_pb2.Image.PixelFormat.PIXEL_FORMAT_RGB_U8, self.done)
            for src in RGB_SOURCES
        ]
        self.img_recorders += [
            LatestImageRecorder(image_client, src, image_pb2.Image.PixelFormat.PIXEL_FORMAT_DEPTH_U16, self.done)
            for src in DEPTH_SOURCES
        ]

        self.latest_instruction = None
        self.latest_instruction_time = None

    def start(self):
        self.robot_state_thread.start()
        for img_rec in self.img_recorders:
            img_rec.start()

    def stop(self):
        self.done.set()
        self.robot_state_thread.join()
        for img_rec in self.img_recorders:
            img_rec.join()

    def add_instruction(self, text: str):
        self.latest_instruction = text
        self.latest_instruction_time = time.time()

    def save_robot_state(self, robot_state_client, done: Event):
        # the dataset is stored as a pkl,
        # containing a list of dicts, where each dict is a snapshot of either
        # the robot state or the image response
        dataset_path = self.root / 'dataset.pkl'
        dataset = []

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

            dataset.append(step_data)

            # save and wait a bit so other threads can run
            with dataset_path.open('wb') as f:
                pickle.dump(dataset, f)
