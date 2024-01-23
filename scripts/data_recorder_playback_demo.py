from pathlib import Path

import rerun as rr

from conq.logging.replay.conq_log_file import ConqLog
from conq.cameras_utils import image_to_opencv

RR_TIMELINE_NAME = "stable_time"
NANOSECONDS_TO_SECONDS = 1e-9


def get_default_log_paths():
    """Do all the path manipulation necessary to point to example log and metadata files

    This is just a convenience function to make the main function less cluttered.
    """
    root_path = Path("/media/big_narstie/datasets/conq_hose_manipulation_raw/")
    exp_name = "conq_hose_manipulation_data_1700079751"
    exp_path = root_path / exp_name
    episode_num = 0

    pkl_path = exp_path / "train" / f"episode_{episode_num}.pkl"
    metadata_path = exp_path / "metadata.json"

    return pkl_path, metadata_path


def main(pkl_path: Path, metadata_path: Path):
    log = ConqLog(pkl_path, metadata_path)

    rr.init("data_recorder_playback_rerun", spawn=True)
    rr.set_time_seconds(RR_TIMELINE_NAME, log.get_t_start())

    # Specify the camera sources you want to pull images from.
    rgb_sources = [
        "frontright_fisheye_image", "frontleft_fisheye_image", "left_fisheye_image",
        "right_fisheye_image", "hand_color_image"
    ]
    depth_sources = ["frontleft_depth_in_visual_frame", "frontright_depth_in_visual_frame"]

    for packet in log.msg_packet_iterator(rate_limit_hz=2):
        # If wishing to access the robot state recorded in this message:
        # state = step['robot_state']
        # snapshot = state.kinematic_state.transforms_snapshot
        # timestamp = state.kinematic_state.acquisition_timestamp
        # print(timestamp)
        # Use snapshot to get the transforms.

        for source_name, res in packet.image_iterator(rgb_sources, depth_sources):
            timestamp_secs = NANOSECONDS_TO_SECONDS * res.shot.acquisition_time.ToNanoseconds()

            # Sets the publication time for the rerun message.
            rr.set_time_seconds(RR_TIMELINE_NAME, timestamp_secs)

            img_np = image_to_opencv(res, auto_rotate=True)
            rr.log(source_name, rr.Image(img_np))

    print("Done with rerun playback.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Demo of viewing a Conq log.")

    parser.add_argument(
        "--log-path",
        "-l",
        type=Path,
        help=("Path to the log file you wish to view. Optional, defaults to a log stored on Big "
              "Narstie for demonstration purposes."))
    parser.add_argument(
        "--metadata-path",
        "-m",
        type=Path,
        help="Path to the metadata file corresponding to the given log file. Optional.")

    args = parser.parse_args()
    log_path = args.log_path
    metadata_path = args.metadata_path
    is_log_path_specifed = log_path is not None
    is_metadata_path_specified = metadata_path is not None

    if is_log_path_specifed and is_metadata_path_specified:
        log_path = Path(log_path)
        metadata_path = Path(metadata_path)
    elif (not is_log_path_specifed) and (not is_metadata_path_specified):
        # Default to example log playback for demo purposes.
        log_path, metadata_path = get_default_log_paths()

        if not log_path.exists():
            raise FileNotFoundError(f"Couldn't find log file '{log_path}'. Is Big Narstie mounted?")
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Couldn't find metadata file '{metadata_path}'. Is Big Narstie mounted?")
    elif (not is_log_path_specifed) and is_metadata_path_specified:
        raise RuntimeError("Must specify log file if providing metadata file.")

    main(log_path, metadata_path)
