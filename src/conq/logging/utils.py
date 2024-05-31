import numpy as np
from scipy.spatial.transform import Rotation as R

def parse_image_response(image_response):
    camera_intrinsics = None
    camera_translation = None
    camera_rotation_matrix = None
    camera_source_name = None
    camera_sensor_frame_name = None
    
    # Extracting the camera source name
    camera_source_name = image_response.source.name
    
    # Extracting the intrinsic matrix
    intrinsics = image_response.source.pinhole.intrinsics
    fx = intrinsics.focal_length.x
    fy = intrinsics.focal_length.y
    cx = intrinsics.principal_point.x
    cy = intrinsics.principal_point.y
    
    camera_intrinsics = np.array([[fx, 0, cx],
                                  [0, fy, cy],
                                  [0, 0, 1]])
    
    # Extracting the sensor frame name and the extrinsic matrix
    transforms_snapshot = image_response.shot.transforms_snapshot
    for edge in transforms_snapshot.child_to_parent_edge_map:
        if edge.key == camera_source_name: # TODO: Need to make this common
            transform = edge.value.parent_tform_child
            position = transform.position
            rotation = transform.rotation
            
            # Translation vector
            camera_translation = np.array([position.x, position.y, position.z])
            
            # Convert quaternion to rotation matrix
            quaternion = [rotation.x, rotation.y, rotation.z, rotation.w]
            r = R.from_quat(quaternion)
            camera_rotation_matrix = r.as_matrix()
            
            # Sensor frame name
            camera_sensor_frame_name = edge.value.parent_frame_name
            break
    
    # Compose the extrinsic matrix
    if camera_translation is not None and camera_rotation_matrix is not None:
        camera_extrinsic = np.eye(4)
        camera_extrinsic[:3, :3] = camera_rotation_matrix
        camera_extrinsic[:3, 3] = camera_translation
    else:
        camera_extrinsic = None
    
    return {
        "camera_intrinsic_matrix": camera_intrinsics,
        "camera_extrinsic_matrix": camera_extrinsic,
        "camera_source_name": camera_source_name,
        "camera_sensor_frame_name": camera_sensor_frame_name
    }