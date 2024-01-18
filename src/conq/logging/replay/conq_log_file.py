from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import pickle

from conq.logging.replay.message_packet import MessagePacket


class ConqLog:

    def __init__(self, log_path: Path, metadata_path: Path):
        self.log_path: Path = self._verify_path_exists(log_path)
        self.metadata_path: Path = self._verify_path_exists(metadata_path)
        self.metadata: Dict[str, Any] = self._read_metadata()
        self.log_data: List[MessagePacket] = self._read_log()

        print(f"Duration of log: {self.get_t_end() - self.get_t_start():.2f} seconds")

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
