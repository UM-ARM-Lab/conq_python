import numpy as np
import rerun as rr
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import get_a_tform_b, VISION_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME, \
    GROUND_PLANE_FRAME_NAME, HAND_FRAME_NAME


def viz_common_frames(snapshot):
    body_in_vision = get_a_tform_b(snapshot, VISION_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME)
    gpe_in_vision = get_a_tform_b(snapshot, VISION_FRAME_NAME, GROUND_PLANE_FRAME_NAME)
    hand_in_vision = get_a_tform_b(snapshot, VISION_FRAME_NAME, HAND_FRAME_NAME)
    rr_tform('body', body_in_vision)
    rr_tform('gpe', gpe_in_vision)
    rr_tform('hand', hand_in_vision)
    rr.log_transform3d(f'frames/odom', rr.Translation3D([0, 0, 0]))


def rr_tform(child_frame: str, tform: math_helpers.SE3Pose):
    translation = np.array([tform.position.x, tform.position.y, tform.position.z])
    rot_mat = tform.rotation.to_matrix()
    rr.log_transform3d(f'frames/{child_frame}', rr.TranslationAndMat3(translation, rot_mat))
