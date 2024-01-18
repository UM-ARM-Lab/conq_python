from typing import Dict, Any, List, Optional
import pickle
from pathlib import Path
import json

import rerun as rr

from conq.cameras_utils import image_to_opencv

RR_TIMELINE_NAME = "stable_time"
NANOSECONDS_TO_SECONDS = 1e-9


class ImageSourceUnavailbleError(Exception):
    pass


class MessagePacket:

    def __init__(self, packet_raw: Dict[str, Any], rgb_sources_available: List[str],
                 depth_sources_available: List[str]):
        self.rgb_sources_available: List[str] = rgb_sources_available
        self.depth_sources_available: List[str] = depth_sources_available
        self.timestamp: float = packet_raw["time"]
        self.robot_state = packet_raw["robot_state"]
        self.images: Dict[str, Any] = packet_raw["images"]

    def image_iterator(self,
                       rgb_sources: Optional[List[str]] = None,
                       depth_sources: Optional[List[str]] = None):
        """Provides iterator for images in packet with optional filtering for camera sources"""
        rgb_sources = self.rgb_sources_available if rgb_sources is None else rgb_sources
        depth_sources = self.depth_sources_available if depth_sources is None else depth_sources

        self._verify_rgb_sources(rgb_sources)
        self._verify_depth_sources(depth_sources)

        # There won't be name conflicts here since each image requires a unique name (thanks to
        # dictionary storage).
        img_srcs_requested = rgb_sources + depth_sources

        # TODO: Filter images by acquisition time?
        for src, img_msg in self.images.items():
            if src in img_srcs_requested:
                yield (src, img_msg)

    def _verify_rgb_sources(self, sources: Optional[List[str]]):
        self._verify_img_sources(sources, self.rgb_sources_available)

    def _verify_depth_sources(self, sources: Optional[List[str]]):
        self._verify_img_sources(sources, self.depth_sources_available)

    def _verify_img_sources(self, sources_requested: Optional[List[str]], sources_avail: List[str]):
        """Verifies the requested image sources are present in the log"""
        if sources_requested is None:
            return

        sources_requested = set(sources_requested)
        sources_avail = set(sources_avail)
        sources_unavail = sources_requested - sources_avail
        if len(sources_unavail) != 0:
            raise ImageSourceUnavailbleError(
                f"Requested image sources that are not in log: {sources_unavail}")


class ConqLogFile:

    def __init__(self, log_path: Path, metadata_path: Path):
        self.log_path: Path = self._verify_path_exists(log_path)
        self.metadata_path: Path = self._verify_path_exists(metadata_path)
        self.metadata: Dict[str, Any] = self._read_metadata()
        self.log_data: List[MessagePacket] = self._read_log()

    def get_available_rgb_sources(self) -> List[str]:
        return self.metadata["rgb_sources"]

    def get_available_depth_sources(self) -> List[str]:
        return self.metadata["depth_sources"]

    def get_t_start(self) -> float:
        """Returns log start time as a Unix timestamp float"""
        return self.log_data[0].timestamp

    def get_t_end(self) -> float:
        """Returns log end time as a Unix timestamp float"""
        return self.log_data[-1].timestamp

    def msg_packet_iterator(self, rate_limit_hz: Optional[float] = None):
        """Generator to iterate through messages, optionally rate-limited

        NOTE: This rate-limits based on the full message timestamp. NOT the per-image capture time.
        This shouldn't be an issue, though.
        """
        # Set the "previous message time" to something far in the past.
        t_prev = self.get_t_start() - 1e5
        rate_limit_secs = 1. / rate_limit_hz if rate_limit_hz is not None else None
        for packet in self.log_data:
            t_diff = packet.timestamp - t_prev

            # Catch if anything is weird with the log, i.e. if the previous message's time stamp is
            # in the future compared to this message.
            if t_diff < 0:
                raise RuntimeError("Detected message packets are out of order.")

            # Skip this message if it's too soon, given the supplied rate limit.
            if (rate_limit_secs is not None) and (t_diff < rate_limit_secs):
                continue

            t_prev = packet.timestamp
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

        packets = []
        for packet_raw in dat:
            packets.append(
                MessagePacket(packet_raw, self.get_available_rgb_sources(),
                              self.get_available_depth_sources()))

        return packets


def main():
    rr.init("data_recorder_playback_rerun", spawn=True)

    root_path = Path("/media/big_narstie/datasets/conq_hose_manipulation_raw/")

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

    rgb_sources_all = log.get_available_rgb_sources()
    depth_sources_all = log.get_available_depth_sources()

    print("Duration of log in seconds:", log.get_t_end() - log.get_t_start())

    rr.set_time_seconds(RR_TIMELINE_NAME, log.get_t_start())

    rgb_sources = [
        # "frontright_fisheye_image",
        # "frontleft_fisheye_image",
        "left_fisheye_image",
        "right_fisheye_image"
    ]
    depth_sources = ["frontleft_depth_in_visual_frame", "frontright_depth_in_visual_frame"]

    for packet in log.msg_packet_iterator(rate_limit_hz=1):
        # state = step['robot_state']
        # snapshot = state.kinematic_state.transforms_snapshot
        # timestamp = state.kinematic_state.acquisition_timestamp
        # print(timestamp)
        # use snapshot to get the transforms

        for source_name, res in packet.image_iterator(rgb_sources, depth_sources):

            timestamp = 1e-9 * res.shot.acquisition_time.ToNanoseconds()

            # Sets the publication time for the rerun message.
            rr.set_time_seconds(RR_TIMELINE_NAME, timestamp)

            # topic_name = f"rgb_{src}"
            # topic_name = source_name

            # if is_rgb:
            #     rgb_np = image_to_opencv(res)
            #     rr.log(topic_name, rr.Image(rgb_np))
            # elif is_depth:
            #     depth_np = image_to_opencv(res)
            #     rr.log(topic_name, rr.Image(depth_np))
            img_np = image_to_opencv(res)
            rr.log(source_name, rr.Image(img_np))


if __name__ == "__main__":
    main()
