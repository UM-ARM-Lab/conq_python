from typing import TYPE_CHECKING, Dict, Any, Optional, List

from conq.logging.exceptions import ImageSourceUnavailableError

if TYPE_CHECKING:
    from bosdyn.api.robot_state_pb2 import RobotState
    from bosdyn.api.graph_nav.nav_pb2 import Localization


class MessagePacket:

    def __init__(self, packet_raw: Dict[str, Any], rgb_sources_available: Optional[List[str]],
                 depth_sources_available: Optional[List[str]]):
        self.rgb_sources_available: Optional[List[str]] = rgb_sources_available
        self.depth_sources_available: Optional[List[str]] = depth_sources_available
        self.timestamp: float = packet_raw["time"]
        self.robot_state: RobotState = packet_raw["robot_state"]
        self.images: Dict[str, Any] = packet_raw["images"]

        # These are optionally recorded.
        self.localization: Optional[Localization] = packet_raw.get("localization", None)
        self.is_lost: Optional[bool] = packet_raw.get("is_lost", None)

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

    def _verify_img_sources(self, sources_requested: Optional[List[str]],
                            sources_avail: Optional[List[str]]):
        """Verifies the requested image sources are present in the log

        If no source requests were provided or if no metadata is available on available sources,
        that's fine.
        """
        if (sources_requested is None) or (sources_avail is None):
            return

        sources_requested = set(sources_requested)
        sources_avail = set(sources_avail)
        sources_unavail = sources_requested - sources_avail
        if len(sources_unavail) != 0:
            raise ImageSourceUnavailableError(
                f"Requested image sources that are not in log: {sources_unavail}")
