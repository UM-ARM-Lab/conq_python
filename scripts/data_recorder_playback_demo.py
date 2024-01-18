from pathlib import Path

import rerun as rr

from conq.logging.replay.conq_log_file import ConqLog
from conq.cameras_utils import image_to_opencv

RR_TIMELINE_NAME = "stable_time"
NANOSECONDS_TO_SECONDS = 1e-9


def get_log_paths():
    """Do all the path manipulation necessary to point to log and metadata files

    This is just a convenience function to make the main function less cluttered.
    """
    root_path = Path("/media/big_narstie/datasets/conq_hose_manipulation_raw/")

    exp_name = "conq_hose_manipulation_data_1700079751"
    # exp_name = "conq_hose_manipulation_data_1700080718"
    # exp_name = "conq_hose_manipulation_data_1700081021"
    # exp_name = "conq_hose_manipulation_data_1700082856"
    exp_path = root_path / exp_name

    episode_num = 0
    # episode_num = 6
    # episode_num = 8

    pkl_path = exp_path / "train" / f"episode_{episode_num}.pkl"
    metadata_path = exp_path / "metadata.json"

    return pkl_path, metadata_path


def main():
    rr.init("data_recorder_playback_rerun", spawn=True)

    pkl_path, metadata_path = get_log_paths()
    log = ConqLog(pkl_path, metadata_path)

    rr.set_time_seconds(RR_TIMELINE_NAME, log.get_t_start())

    # Specify the camera sources you want to pull images from.
    rgb_sources = [
        "frontright_fisheye_image", "frontleft_fisheye_image", "left_fisheye_image",
        "right_fisheye_image", "hand_color_image"
    ]
    depth_sources = ["frontleft_depth_in_visual_frame", "frontright_depth_in_visual_frame"]

    for packet in log.msg_packet_iterator(rate_limit_hz=2):
        # state = step['robot_state']
        # snapshot = state.kinematic_state.transforms_snapshot
        # timestamp = state.kinematic_state.acquisition_timestamp
        # print(timestamp)
        # use snapshot to get the transforms

        for source_name, res in packet.image_iterator(rgb_sources, depth_sources):
            timestamp_secs = NANOSECONDS_TO_SECONDS * res.shot.acquisition_time.ToNanoseconds()

            # Sets the publication time for the rerun message.
            rr.set_time_seconds(RR_TIMELINE_NAME, timestamp_secs)

            img_np = image_to_opencv(res, auto_rotate=True)
            rr.log(source_name, rr.Image(img_np))

    print("Done with rerun playback.")


if __name__ == "__main__":
    main()
