from typing import Dict, Any, List, Optional
import pickle
from pathlib import Path
import json

import rerun as rr

from conq.cameras_utils import image_to_opencv

RR_TIMELINE_NAME = "stable_time"

class ConqLogFile:
    def __init__(self, log_path: Path, metadata_path: Path):
        self.log_path: Path = self._verify_path_exists(log_path)
        self.metadata_path: Path = self._verify_path_exists(metadata_path)
        self.log_data: Dict[str, Any] = self._read_log()
        self.metadata: Dict[str, Any] = self._read_metadata()

    def get_available_rgb_sources(self) -> List[str]:
        return self.metadata["rgb_sources"]

    def get_available_depth_sources(self) -> List[str]:
        return self.metadata["depth_sources"]

    def get_t_start(self) -> float:
        """Returns log start time as a Unix timestamp float"""
        return self.log_data[0]["time"]

    def get_t_end(self) -> float:
        """Returns log end time as a Unix timestamp float"""
        return self.log_data[-1]["time"]

    def msg_packet_iterator(self, rate_limit_secs: Optional[float] = None):
        """Generator to iterate through messages, optionally rate-limited

        NOTE: This rate-limits based on the full message timestamp. NOT the per-image capture time.
        This shouldn't be an issue, though.
        """
        # Set the "previous message time" to something far in the past.
        t_prev = self.get_t_start() - 1e5
        for packet in self.log_data:
            t_packet = packet["time"]
            t_diff = t_packet - t_prev

            # Catch if anything is weird with the log, i.e. if the previous message's time stamp is
            # in the future compared to this message.
            if t_diff < 0:
                raise RuntimeError("Detected message packets are out of order.")

            # Skip this message if it's too soon, given the supplied rate limit.
            if (rate_limit_secs is not None) and (t_diff < rate_limit_secs):
                continue

            yield packet


    def _verify_path_exists(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(f"Didn't find file: {path}")
        return path

    def _read_metadata(self) -> Dict[str, Any]:
        with self.metadata_path.open('rb') as f:
            met = json.load(f)
        return met

    def _read_log(self) -> Dict[str, Any]:
        with self.log_path.open('rb') as f:
            dat = pickle.load(f)
        return dat

def main():
    rr.init("data_recorder_playback_rerun", spawn=True)

    root_path = Path(
        "/media/big_narstie/datasets/conq_hose_manipulation_raw/")

    # exp_name = "conq_hose_manipulation_data_1700079751"
    # exp_name = "conq_hose_manipulation_data_1700080718"
    exp_name = "conq_hose_manipulation_data_1700081021"
    exp_path = root_path / exp_name

    metadata_path = exp_path / "metadata.json"
    assert metadata_path.exists()

    # episode_num = 0
    episode_num = 8

    pkl_path = exp_path / "train" / f"episode_{episode_num}.pkl"
    # assert pkl_path.exists()

    log = ConqLogFile(pkl_path, metadata_path)

    # with metadata_path.open('rb') as f:
    #     metadata = json.load(f)

    rgb_sources_all = log.get_available_rgb_sources()
    depth_sources_all = log.get_available_depth_sources()

    # with pkl_path.open('rb') as f:
    #     data = pickle.load(f)

    # t_start = data[0]["time"]
    # t_end = data[-1]["time"]
    print("Duration of log in seconds:", log.get_t_end() - log.get_t_start())

    rr.set_time_seconds(RR_TIMELINE_NAME, log.get_t_start())

    rgb_sources = [
        # "frontright_fisheye_image",
        # "frontleft_fisheye_image",
        "left_fisheye_image",
        "right_fisheye_image"
    ]
    depth_sources = [
        "frontleft_depth_in_visual_frame",
        "frontright_depth_in_visual_frame"
    ]

    # data_used = data[::10]
    # data_used = data

    # for step in data_used:
    for step in log.msg_packet_iterator():
        # state = step['robot_state']
        # snapshot = state.kinematic_state.transforms_snapshot
        # timestamp = state.kinematic_state.acquisition_timestamp
        # print(timestamp)
        # use snapshot to get the transforms



        for src, res in step['images'].items():
            is_rgb = src in rgb_sources_all
            is_depth = src in depth_sources_all
            if (not is_rgb) and (not is_depth):
                raise RuntimeError("Didn't understand src:", src)

            timestamp = 1e-9 * res.shot.acquisition_time.ToNanoseconds()
            # timestamp = step["time"]

            # Sets the publication time for the rerun message.
            rr.set_time_seconds(RR_TIMELINE_NAME, timestamp)

            # topic_name = f"rgb_{src}"
            topic_name = src

            # if is_rgb:
            #     rgb_np = image_to_opencv(res)
            #     rr.log(topic_name, rr.Image(rgb_np))
            # elif is_depth:
            #     depth_np = image_to_opencv(res)
            #     rr.log(topic_name, rr.Image(depth_np))
            img_np = image_to_opencv(res)
            rr.log(topic_name, rr.Image(img_np))


if __name__ == "__main__":
    main()
