import numpy as np
from pathlib import Path
# VIZ
import rerun as rr
import rerun.blueprint as rrb
# CONQ: Clients
from conq.clients import Clients
from conq.rerun_utils import viz_common_frames, rr_tform
from conq.cameras_utils import get_color_img

#BOSDYN: Helpers
from bosdyn.client.frame_helpers import get_a_tform_b,VISION_FRAME_NAME, HAND_FRAME_NAME,GRAV_ALIGNED_BODY_FRAME_NAME

# BOSDYN
from bosdyn.client.image import depth_image_data_to_numpy


SOURCES =  [
            ["hand_depth_in_hand_color_frame","hand_color_image"],
            ["frontleft_depth_in_visual_frame","frontleft_fisheye_image"],
            ["frontright_depth_in_visual_frame","frontright_fisheye_image"],
            ["left_depth_in_visual_frame","left_fisheye_image"],
            ["right_depth_in_visual_frame","right_fisheye_image"],
            ["back_depth_in_visual_frame","back_fisheye_image"],
            ]

class ConqLogger:
    dir_path:Path
    sources:list
    clients:Clients

    def __init__(self,dir_path:Path,clients):
        self.dir_path = dir_path
        # camera_sources
        # camera_sources: Intrinsics, Extrinsics
        self.image_client = clients.image
        self.state_client = clients.state
        self.robot = clients.robot
        self.origin = "robot/"
        # self.sources = self.image_client.list_image_sources()
        # Sources need to be made in pairs (RGB, Depth:aligned with RGB)
        self.sources = [
            ["hand_depth_in_hand_color_frame","hand_color_image"],
            ["frontleft_depth_in_visual_frame","frontleft_fisheye_image"],
            ["frontright_depth_in_visual_frame","frontright_fisheye_image"],
            ["left_depth_in_visual_frame","left_fisheye_image"],
            ["right_depth_in_visual_frame","right_fisheye_image"],
            ["back_depth_in_visual_frame","back_fisheye_image"],
            ]
        self.ee_pos = []

    def log(self, all = True):
        """Log all the variables"""
        self.log_camera()
        self.log_state()
        self.log_action()

    def log_camera(self):
        for source_pair in self.sources:
            print("______________________________\n")
            print("Source being logged: ",source_pair)
            images_proto = self.image_client.get_image_from_sources(source_pair)

            RGB_NP, _ = get_color_img(self.image_client, source_pair[1])
            RGB_NP = np.array(RGB_NP, dtype=np.uint8)

            DEPTH_NP = np.float32(_depth_image_data_to_numpy(image_response=images_proto[0]))
            
            sensor_name = images_proto[1].shot.frame_name_image_sensor
            BODY_T_VISION = (get_a_tform_b(images_proto[0].shot.transforms_snapshot, "body", sensor_name)).to_matrix() # 4x4
            print("Transformation Matrix: ", BODY_T_VISION)
            print("______________________________\n")
            # RGB and Depth (aligned with RGB) have the same intrinsics and extrinsics
            source_name = sensor_name.split('_')[0]
            rr.log(
                f"{self.origin}camera/{source_name}",
                rr.Transform3D(
                    translation=BODY_T_VISION[:3,3],
                    mat3x3=BODY_T_VISION[:3,:3],
                    from_parent=True,
                ),
                timeless=True,
            )
            
            rr.log(
                f"{self.origin}camera/{source_name}",
                rr.Pinhole(
                    resolution=[images_proto[1].source.cols, images_proto[1].source.rows], #(640,480)
                    focal_length=[images_proto[1].source.pinhole.intrinsics.focal_length.x, images_proto[1].source.pinhole.intrinsics.focal_length.y],
                    principal_point=[images_proto[1].source.pinhole.intrinsics.principal_point.x, images_proto[1].source.pinhole.intrinsics.principal_point.y], # (320,240)
                ),
            )
            depth_scale = images_proto[1].source.depth_scale # 1000
            #DEPTH_NP[:,-1] = DEPTH_NP[:,-1] / depth_scale
            
            rr.log(f"{self.origin}camera/{source_name}/rgb", rr.Image(RGB_NP))
            rr.log(f"{self.origin}camera/{source_name}/depth", rr.DepthImage(DEPTH_NP,meter = depth_scale))
        
    def log_state(self):
        "Logs known/commonly used frames over time"
        state = self.state_client.get_robot_state()
        viz_common_frames(state.kinematic_state.transforms_snapshot)

        ee_in_vision = get_a_tform_b(state.kinematic_state.transforms_snapshot, VISION_FRAME_NAME, HAND_FRAME_NAME)
        self.ee_pos.append([ee_in_vision.position.x, ee_in_vision.position.y, ee_in_vision.position.z])
        rr.log(f'{self.origin}action/arm/cartesian/positions',rr.Points3D(self.ee_pos, colors=[1.0, 0, 0]))

    def log_action(self):
        "Logs arm joint angles and velocities"
        state = self.state_client.get_robot_state()
        
        joint_positions = [js.position.value for js in state.kinematic_state.joint_states]
        joint_velocities = [js.velocity.value for js in state.kinematic_state.joint_states]

        for i, joint_position in enumerate(joint_positions[-6:]):
            rr.log(f'{self.origin}action/arm/joint/positions/{i}', rr.Scalar(joint_position))

        for i, joint_velocity in enumerate(joint_velocities[-6:]):
            rr.log(f'{self.origin}action/arm/joint/velocities/{i}', rr.Scalar(joint_velocity))

    def save_logs(self):
        pass

def extract_prefix(string):
        return string.split('_')[0]

def process_sources(root,sources):
    source_names = []
    for source_list in sources:
        # Extract prefix from the first element of each nested list
        prefix = extract_prefix(source_list[0])
        source_names.append(prefix)
    
    origins = [f"{root}camera/{name}" for name in source_names]
    return origins

def get_blueprint(root):
    origins = process_sources(root,SOURCES)
    print(origins)
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Spatial3DView(name="robot view", origin="/", contents=["/**"]),
                rrb.Horizontal(
                    *(rrb.Spatial2DView(
                        name=f"camera_{org}",
                        origin=org,
                    ) for org in origins),
                ),
            ),
            rrb.Tabs(
                rrb.Vertical(
                    contents=[
                        rrb.TimeSeriesView(
                            name=f"Joint Position {i}",
                            origin=f"{root}action/arm/joint/positions/{i}",
                        )
                        for i in range(6)
                    ],name="Joint Position"
                    ),
                rrb.Vertical(
                    contents=[
                        rrb.TimeSeriesView(
                            name=f"Joint Velocity {i}",
                            origin=f"{root}action/arm/joint/velocities/{i}",
                        )
                        for i in range(6)
                    ],name="Joint Velocity"
                    ),
                rrb.Vertical(
                    contents=[
                        rrb.TimeSeriesView(
                            name=f"Gripper Position {i}",
                            origin=f"{root}action/arm/cartesian/positions/{i}",
                        )
                        for i in range(3)
                    ],name="Gripper Position"
                    ),
                rrb.Vertical(
                    contents=[
                        rrb.TimeSeriesView(
                            name=f"Gripper Velocity {i}",
                            origin=f"{root}action/arm/cartesian/velocities/{i}",
                        )
                        for i in range(3)
                    ],name="Gripper Velocity"
                    ),
            )
        ),
        rrb.BlueprintPanel(expanded=False),
        rrb.SelectionPanel(expanded=False),
        rrb.TimePanel(expanded=False),
        auto_space_views=False,
    )
    return blueprint