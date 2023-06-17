import json
from pathlib import Path
from typing import Dict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from cdcpd_torch.core.deformable_object_configuration import RopeConfiguration
from cdcpd_torch.core.tracking_map import TrackingMap
from cdcpd_torch.data_utils.img_cloud_utils import imgs_to_clouds_np, cloud_to_img_np
from cdcpd_torch.data_utils.types.point_cloud import PointCloud
from cdcpd_torch.modules.cdcpd_module_arguments import CDCPDModuleArguments
from cdcpd_torch.modules.cdcpd_network import CDCPDModule
from cdcpd_torch.modules.cdcpd_parameters import CDCPDParamValues
from cdcpd_torch.modules.post_processing.configuration import PostProcConfig, PostProcModuleChoice
from conq.exceptions import DetectionError

NUM_POINTS = 15
METERS_TO_MILLIMETERS = 1e3

INTRINSIC_INFO = {
    "cols": 640,
    "rows": 480,
    "pinhole": {
        "intrinsics": {
            "focal_length": {
                "x": 552.02910121610671,
                "y": 552.02910121610671
            },
            "principal_point": {
                "x": 320,
                "y": 240
            }
        }
    }
}

fx = INTRINSIC_INFO["pinhole"]["intrinsics"]["focal_length"]["x"]
fy = INTRINSIC_INFO["pinhole"]["intrinsics"]["focal_length"]["y"]
px = INTRINSIC_INFO["pinhole"]["intrinsics"]["principal_point"]["x"]
py = INTRINSIC_INFO["pinhole"]["intrinsics"]["principal_point"]["y"]
INTRINSIC = np.array((
    (fx, 0, px),
    (0, fy, py),
    (0, 0, 1)
), dtype=float)


def load_predictions(pred_filepath):
    predictions_path = Path(pred_filepath)
    if predictions_path.exists():
        with predictions_path.open("r") as f:
            predictions = json.load(f)
    else:
        raise RuntimeError("Couldn't find predictions path:", predictions_path)

    return predictions


def get_masks_dict(predictions, img, rope_classes=None, obstacle_class_name="battery"):
    # create masks based on the polygons for each class
    if rope_classes is None:
        rope_classes = ["vacuum_hose", "vacuum_neck"]
    rope_masks = []
    obstacle_masks = []
    rope_polygons = []
    obstacle_polygons = []
    for pred in predictions:
        class_name = pred["class"]

        points = pred["points"]
        points = np.array([(p['x'], p['y']) for p in points], dtype=int)
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)

        if class_name in rope_classes:
            rope_polygons.append(points)
            rope_masks.append(mask)
        elif class_name == obstacle_class_name:
            obstacle_polygons.append(points)
            obstacle_masks.append(mask)

    if len(rope_masks) == 0:
        raise DetectionError("No rope detected")

    masks = {
        "rope": rope_masks,
        "obstacle": obstacle_masks
    }

    return masks


def show_img_and_mask(img, mask, mask_name):
    fig, ax = plt.subplots(ncols=2)
    ax[0].imshow(img)
    ax[1].imshow(mask)
    ax[1].set_title(mask_name)
    # fig.show()


def combine_masks(img, masks):
    combined_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for mask in masks:
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    return combined_mask


## CDCPD Helper Functions
## ----------------------

def find_rope_start_end_points(cloud_filtered: PointCloud):
    # Find the point of the "rope" closest to the far upper left of the robot and the point closeset to
    # the far lower right of the robot for the start and end points.
    cloud_pts = cloud_filtered.xyz
    cloud_mean_z_val = cloud_filtered.xyz[2, :].mean()
    upper_left_pt = np.array((-1e5, -1e5, cloud_mean_z_val)).reshape(3, 1)
    lower_right_pt = np.array((1e5, 1e5, cloud_mean_z_val)).reshape(3, 1)

    cloud_upper_left_norms = np.linalg.norm(cloud_pts - upper_left_pt, axis=0)
    cloud_lower_right_norms = np.linalg.norm(cloud_pts - lower_right_pt, axis=0)

    cloud_upper_left_pt_idx = np.argmin(cloud_upper_left_norms)
    cloud_lower_right_pt_idx = np.argmin(cloud_lower_right_norms)

    cloud_upper_left_pt = cloud_pts[:, cloud_upper_left_pt_idx]
    cloud_lower_right_pt = cloud_pts[:, cloud_lower_right_pt_idx]

    return cloud_upper_left_pt, cloud_lower_right_pt


def generate_multiple_start_end_points(cloud_filtered: PointCloud):
    cloud_pts = cloud_filtered.xyz
    cloud_mean_z_val = cloud_filtered.xyz[2, :].mean()

    cloud_mins = cloud_pts.min(axis=1)
    cloud_maxes = cloud_pts.max(axis=1)


    # Generate configurations that go from:
    # - top left to bottom right
    # - top middle to bottom middle
    # - top right to bottom left
    # - middle left to middle right.
    top = cloud_mins[0]
    bottom = cloud_maxes[0]
    left = cloud_mins[1]
    right = cloud_maxes[1]
    # top = cloud_mins[1]
    # bottom = cloud_maxes[1]
    # left = cloud_mins[0]
    # right = cloud_maxes[0]
    middle_vert = (top + bottom) / 2
    middle_horiz = (left + right) / 2
    # top_left = cloud_mins[:2]
    # top_right = np.concatenate((cloud_mins[0], cloud_maxes[1]))

    start_pts = np.zeros((3, 4))
    end_pts = np.zeros((3, 4))

    # set the z values to the mean of all z values.
    start_pts[2, :] = cloud_mean_z_val
    end_pts[2, :] = cloud_mean_z_val

    idx = 0
    # Top left to bottom right
    start_pts[0, idx] = top
    start_pts[1, idx] = left
    end_pts[0, idx] = bottom
    end_pts[1, idx] = right
    idx += 1

    # Top middle to bottom middle
    start_pts[0, idx] = top
    start_pts[1, idx] = middle_horiz
    end_pts[0, idx] = bottom
    end_pts[1, idx] = middle_horiz
    idx += 1

    # Top right to bottom left
    start_pts[0, idx] = top
    start_pts[1, idx] = right
    end_pts[0, idx] = bottom
    end_pts[1, idx] = left
    idx += 1

    # middle left to middle right
    start_pts[0, idx] = middle_vert
    start_pts[1, idx] = left
    end_pts[0, idx] = middle_vert
    end_pts[1, idx] = right
    idx += 1

    return start_pts, end_pts


def setup_tracking(rope_start_pt: np.ndarray, rope_end_pt: np.ndarray):
    # RopeConfiguration just makes template a straight line between the two points.
    # NOTE: I don't even think that the max_rope_length is used in *just* CPD. I think just in
    # post-processing
    max_rope_length = 1.25 * np.abs(rope_end_pt - rope_start_pt)
    rope_config = RopeConfiguration(
        NUM_POINTS,
        max_rope_length,
        rope_start_position=torch.from_numpy(rope_start_pt).to(torch.double),
        rope_end_position=torch.from_numpy(rope_end_pt).to(torch.double)
    )
    rope_config.initialize_tracking()

    tracking_map = TrackingMap()
    tracking_map.add_def_obj_configuration(rope_config)

    return rope_config, tracking_map


def setup_cdcpd_module(rope_config: RopeConfiguration):
    # turning off post-processing to see if CPD is sufficient.
    postproc_config = PostProcConfig(module_choice=PostProcModuleChoice.NONE)

    # The tracking isn't quite converging to what we want it to, likely due to dynamics regularization.
    # Trying to turn that down to get better agreement here.
    param_vals = CDCPDParamValues()
    # This seemed to do the job. Hopefully it doesn't explode or get really poor tracking when using
    # poor initializations.
    param_vals.zeta_.val = 0.05

    cdcpd_module = CDCPDModule(
        rope_config,
        postprocessing_option=postproc_config,
        param_vals=param_vals,
        # debug=True
    ).eval()

    return cdcpd_module


def project_tracking_results_to_image_coords(cdcpd_output: torch.Tensor):
    cloud_out = PointCloud(cdcpd_output)
    # Guarantee we're giving the `cloud_to_img_np` function a numpy array.
    cloud_out.to_numpy()
    vertex_uv_coords = cloud_to_img_np(cloud_out, INTRINSIC)

    return vertex_uv_coords


def draw_results(rgb_img: np.ndarray, rope_mask_full: np.ndarray,
                 vertex_uv_coords: np.ndarray, saved_name=None):
    num_uv_coords = vertex_uv_coords.shape[1]
    img_viz = rgb_img.copy()

    img_viz[rope_mask_full.astype(bool), 2] = 255

    pt_color = (255, 0, 0)
    for i in range(num_uv_coords):
        pt_coords = vertex_uv_coords[:, i]
        cv2.circle(img_viz, center=pt_coords, radius=5, color=pt_color, thickness=-1)

    if saved_name:
        fig, ax = plt.subplots()
        ax.imshow(img_viz)
        ax.set_title(saved_name.stem)
        fig.savefig(saved_name)
        fig.close()
        print("Saved results figure to:", saved_name)

    return img_viz


def single_frame_planar_cdcpd(rgb_np: np.ndarray, predictions: Dict,
                              do_visualization=False, saved_fig_name=None):
    """Does full CDCPD single frame prediction given input images, predictions, and masks"""
    masks_dict = get_masks_dict(predictions, rgb_np)
    rope_mask_full = combine_masks(rgb_np, masks_dict['rope'])
    depth_np = np.ones((rgb_np.shape[0], rgb_np.shape[1]), dtype=float) * METERS_TO_MILLIMETERS

    if rgb_np.dtype != float:
        rgb_np = rgb_np.astype(float)

    if rgb_np.max() > 1.0:
        rgb_np_normed = rgb_np / 255.

    # Convert images to clouds and downsample.
    cloud_unfiltered, cloud_filtered = imgs_to_clouds_np(rgb_np_normed, depth_np, INTRINSIC,
                                                         rope_mask_full.astype(bool))
    cloud_filtered.downsample(voxel_size=0.02)

    start_pts, end_pts = generate_multiple_start_end_points(cloud_filtered)
    num_generated = start_pts.shape[1]

    best_sigma2 = 1e10
    best_Y_cpd = None
    cloud_filtered.to_torch()
    for i in range(num_generated):
        rope_start_pt = start_pts[:, i]
        rope_end_pt = end_pts[:, i]
        rope_config, tracking_map = setup_tracking(rope_start_pt, rope_end_pt)

        cdcpd_module = setup_cdcpd_module(rope_config)

        inputs = CDCPDModuleArguments(tracking_map, cloud_filtered.xyz)

        outputs = cdcpd_module(inputs)

        Y_cpd_candidate = outputs.get_Y_cpd()
        sigma2 = outputs.get_sigma2_cpd()

        if sigma2 < best_sigma2:
            best_sigma2 = sigma2
            best_Y_cpd = Y_cpd_candidate

    # Project tracking result back to image space coordinates.
    vertex_uv_coords = project_tracking_results_to_image_coords(best_Y_cpd)

    if do_visualization:
        draw_results(rgb_np.astype(np.uint8), rope_mask_full, vertex_uv_coords,
                     saved_name=saved_fig_name)

    return vertex_uv_coords.T
