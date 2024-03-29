from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import pickle

from conq.logging.replay.message_packet import MessagePacket


class ConqLog:

    def __init__(self,
                 log_path: Path,
                 metadata_path: Optional[Path] = None,
                 episode_rate_limit_hz: float = 2):
        """Log file of recorded raw protobuf messages received from Conq for use in playback

        Has some simple useful features like rate-limiting messages for playback.
        """
        self.log_path: Path = self._verify_path_exists(log_path)
        self.metadata_path: Optional[Path] = self._verify_path_exists(metadata_path)
        self.metadata: Optional[Dict[str, Any]] = self._read_metadata()
        self.episode_rate_limit_hz: float = episode_rate_limit_hz
        self.log_data: List[MessagePacket] = self._read_log()

        print(f"Duration of log: {self.get_t_end() - self.get_t_start():.2f} seconds")

    def get_available_rgb_sources(self) -> Optional[List[str]]:
        sources = None
        if self.metadata is not None:
            sources = self.metadata["rgb_sources"]
        return sources

    def get_available_depth_sources(self) -> Optional[List[str]]:
        sources = None
        if self.metadata is not None:
            sources = self.metadata["depth_sources"]
        return sources

    def get_t_start(self) -> float:
        """Returns log start time as a Unix timestamp float"""
        return self.log_data[0].timestamp

    def get_t_end(self) -> float:
        """Returns log end time as a Unix timestamp float"""
        return self.log_data[-1].timestamp

    def msg_packet_iterator(self, rate_limit_hz: Optional[float] = None):
        """Generator to iterate through messages, optionally rate-limited

        You'll probably want to rate-limit using the `ConqLog` rate-limit option instead of this
        iterator's rate-limiting. This saves memory since the `ConqLog` discards messages that come
        at too high of frequency.

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

    def _verify_path_exists(self, path: Optional[Path]):
        if path is None:
            return path

        if not path.exists():
            raise FileNotFoundError(f"Didn't find file: {path}")
        return path

    def _read_metadata(self) -> Dict[str, Any]:
        if self.metadata_path is None:
            return None

        with self.metadata_path.open('rb') as f:
            met = json.load(f)
        return met

    def _read_log(self) -> Dict[str, Any]:
        """Reads all episode pickle files in the log directory"""
        pkl_paths = self.log_path.glob("*.pkl")
        episode_nums = sorted([int(p.stem.split("_")[-1]) for p in pkl_paths])

        packets = []
        t_last = -1e5  # Last timestamp of packet we kept (due to downsampling).
        t_period = 1. / self.episode_rate_limit_hz
        for episode_num in episode_nums:
            try:
                episode_path = self.log_path / f"episode_{episode_num}.pkl"
                with episode_path.open('rb') as f:
                    dat = pickle.load(f)
            except Exception as e:
                print("Couldn't load episode pickle file: ", episode_path)
                print("Skipping this episode due to following exception:")
                print(e)
                continue

            for packet_raw in dat:
                t_packet = packet_raw["time"]
                t_elapsed = t_packet - t_last
                if t_elapsed < t_period:
                    continue
                else:
                    packets.append(
                        MessagePacket(packet_raw, self.get_available_rgb_sources(),
                                      self.get_available_depth_sources()))
                    t_last = packet_raw["time"]

        return packets
