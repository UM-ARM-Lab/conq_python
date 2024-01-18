from pathlib import Path

import rerun as rr

from conq.logging.replay.conq_log_file import ConqLog
from conq.cameras_utils import image_to_opencv

RR_TIMELINE_NAME = "stable_time"
NANOSECONDS_TO_SECONDS = 1e-9
ROOT_PATH = Path("/media/big_narstie/datasets/conq_hose_manipulation_raw/")


def main():
    rr.init("data_recorder_playback_rerun", spawn=True)

    # Do all path manipulation necessary to point to log file and metadata file.
    # exp_name = "conq_hose_manipulation_data_1700079751"
    # exp_name = "conq_hose_manipulation_data_1700080718"
    # exp_name = "conq_hose_manipulation_data_1700081021"
    exp_name = "conq_hose_manipulation_data_1700082856"
    exp_path = ROOT_PATH / exp_name
    # episode_num = 0
    episode_num = 6
    # episode_num = 8
    pkl_path = exp_path / "train" / f"episode_{episode_num}.pkl"
    metadata_path = exp_path / "metadata.json"

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

        # for source_name, res in packet.image_iterator(rgb_sources, depth_sources):
        for source_name, res in packet.image_iterator():
            timestamp_secs = 1e-9 * res.shot.acquisition_time.ToNanoseconds()

            # Sets the publication time for the rerun message.
            rr.set_time_seconds(RR_TIMELINE_NAME, timestamp_secs)

            img_np = image_to_opencv(res, auto_rotate=True)
            rr.log(source_name, rr.Image(img_np))

    print("Done with rerun playback.")


if __name__ == "__main__":
    main()
