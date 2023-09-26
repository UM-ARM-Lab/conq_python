import sys
import time

import bosdyn.client
import bosdyn.client.util
import numpy as np
import rerun as rr
from bosdyn.client.frame_helpers import get_a_tform_b, GROUND_PLANE_FRAME_NAME, BODY_FRAME_NAME
from bosdyn.client.image import ImageClient
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client import math_helpers
from diffcp import SolverError

from arm_segmentation.predictor import Predictor, get_combined_mask
from cdcpd_torch.data_utils.types.point_cloud import PointCloud
from cdcpd_torch.modules.cdcpd_module_arguments import CDCPDModuleArguments
from cdcpd_torch.modules.cdcpd_network import CDCPDModule
from cdcpd_torch.modules.cdcpd_parameters import CDCPDParamValues
from cdcpd_torch.modules.post_processing.configuration import PostProcConfig, PostProcModuleChoice
from conq.cameras_utils import get_color_img, get_depth_img
from conq.exceptions import DetectionError
from conq.perception import project_points_in_gpe, get_gpe_in_cam
from regrasping_demo.cdcpd_hose_state_predictor import setup_tracking


def process_inputs(robot_state_client, image_client, predictor):
    rgb_np, rgb_res = get_color_img(image_client, 'hand_color_image')
    depth_np, depth_res = get_depth_img(image_client, 'hand_depth_in_hand_color_frame')
    predictions = predictor.predict(rgb_np)
    # save_data(rgb_np, depth_np, predictions)
    rope_mask = get_combined_mask(predictions, ['vacuum_hose', 'vacuum_neck'])
    if rope_mask is None:
        raise DetectionError("No rope masks found")
    binary_rope_mask = rope_mask > 0.5

    rr.log_image('cdcpd/rgb', rgb_np)
    rr.log_image('cdcpd/depth', depth_np)
    rr.log_image('cdcpd/rope_mask', rope_mask)

    if not np.any(binary_rope_mask):
        raise DetectionError("No pixels in rope mask > 0.5")
    # Combine masks by adding and clipping the probabilities.

    pixels = np.stack(np.argwhere(binary_rope_mask), 0)

    gpe2cam = get_gpe_in_cam(rgb_res, robot_state_client)
    xyz_in_gpe = project_points_in_gpe(pixels, rgb_res, gpe2cam)

    cam2gpe = gpe2cam.inverse()
    translation = np.array([cam2gpe.position.x, cam2gpe.position.y, cam2gpe.position.z])
    rot_mat = cam2gpe.rotation.to_matrix()
    rr.log_transform3d('world/cam', rr.TranslationAndMat3(translation, rot_mat))

    cloud_filtered = PointCloud(xyz_in_gpe)
    cloud_filtered.downsample(voxel_size=0.02)
    return cloud_filtered


def rr_tform(child_frame: str, tform: math_helpers.SE3Pose):
    translation = np.array([tform.position.x, tform.position.y, tform.position.z])
    rot_mat = tform.rotation.to_matrix()
    rr.log_transform3d(f'world/{child_frame}', rr.TranslationAndMat3(translation, rot_mat))


def main(argv):
    rr.init("hose_tracking")
    rr.connect()

    rr.log_arrow('world/x', [0, 0, 0], [1, 0, 0], color=[1, 0, 0, 1.0], width_scale=0.01)
    rr.log_arrow('world/y', [0, 0, 0], [0, 1, 0], color=[0, 1, 0, 1.0], width_scale=0.01)
    rr.log_arrow('world/z', [0, 0, 0], [0, 0, 1], color=[0, 0, 1, 1.0], width_scale=0.01)

    predictor = Predictor('models/hose_regrasping.pth')

    # Creates client, robot, and authenticates, and time syncs
    sdk = bosdyn.client.create_standard_sdk('continuous_regrasping_hose_demo')
    robot = sdk.create_robot('192.168.80.3')
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()

    assert not robot.is_estopped(), "Robot is estopped!"

    image_client = robot.ensure_client(ImageClient.default_service_name)
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)

    while True:
        try:
            # cloud_filtered = process_inputs(robot_state_client, image_client, predictor)
            # if cloud_filtered.xyz.shape[1] < 100:
            #     print("not enough points!")
            #     time.sleep(5)
            #     continue
            #
            # cloud_filtered.to_numpy()
            # rr.log_points('world/cloud_filtered', cloud_filtered.xyz.T)
            # cloud_filtered.to_torch()
            #
            # Find a good initialization
            # start_pts, end_pts = generate_multiple_start_end_points(cloud_filtered)
            # num_generated = start_pts.shape[1]
            #
            # best_sigma2 = 1e10
            # best_Y_cpd = None
            # best_config = None
            # cloud_filtered.to_torch()
            #
            # for i in range(num_generated):
            #     rope_start_pt = start_pts[:, i]
            #     rope_end_pt = end_pts[:, i]
            #     rope_config, tracking_map = setup_tracking(rope_start_pt, rope_end_pt)
            #
            #     cdcpd_module = setup_cdcpd_module(rope_config)
            #
            #     inputs = CDCPDModuleArguments(tracking_map, cloud_filtered.xyz)
            #
            #     for _ in range(10):
            #         outputs = cdcpd_module(inputs)
            #
            #     Y_cpd_candidate = outputs.get_Y_cpd()
            #     sigma2 = outputs.get_sigma2_cpd()
            #
            #     if sigma2 < best_sigma2:
            #         best_sigma2 = sigma2
            #         best_Y_cpd = Y_cpd_candidate
            #         best_config = rope_config
            # Now run live tracking starting from the best initialiation
            rope_start_pt = np.array([0.7, 0.8, 0])
            rope_end_pt = np.array([1.5, 1.7, 0])
            rope_config, tracking_map = setup_tracking(rope_start_pt, rope_end_pt)
            init_points = tracking_map.form_vertices_cloud().detach().numpy().T
            rr.log_points('world/hose', init_points)
            rr.log_line_strip('cdcpd/hose_line', init_points)
            postproc_config = PostProcConfig(module_choice=PostProcModuleChoice.NONE)
            param_vals = CDCPDParamValues()
            cdcpd_module = CDCPDModule(rope_config, postprocessing_option=postproc_config, param_vals=param_vals).eval()
            while True:
                transforms_body = robot_state_client.get_robot_state().kinematic_state.transforms_snapshot
                body_in_odom = get_a_tform_b(transforms_body, GROUND_PLANE_FRAME_NAME, BODY_FRAME_NAME)
                rr_tform('body', body_in_odom)

                cloud_filtered = process_inputs(robot_state_client, image_client, predictor)
                rr.log_points('world/cloud_filtered', cloud_filtered.xyz.T)
                cloud_filtered.to_torch()
                inputs = CDCPDModuleArguments(tracking_map, cloud_filtered.xyz)
                try:
                    outputs = cdcpd_module(inputs)
                except SolverError as e:
                    print("Solver error")
                    print(e)
                    continue
                cdcpd_output = outputs.get_Y_cpd()
                cloud_out = PointCloud(cdcpd_output)
                cloud_out.to_numpy()
                rr.log_points('world/hose', cloud_out.xyz.T)
                rr.log_line_strip('cdcpd/hose_line', cloud_out.xyz.T)
        except DetectionError:
            print("Detection error")
            time.sleep(5)


if __name__ == '__main__':
    # Checks to see IP address is specified
    if not main(sys.argv[1:]):
        sys.exit(1)
