import numpy as np
if not hasattr(np, 'float'):
    np.float = float

from image_geometry import PinholeCameraModel
from sensor_msgs.msg import CameraInfo
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
from tf_transformations import quaternion_matrix, quaternion_inverse
from builtin_interfaces.msg import Time as RosTime

class HumanLocalizer:
    """
    Localize a human in the ground‐parallel camera frame using:
      - depth image (32FC1)
      - 2D keypoints
      - IMU tilt from ROS TF
    """

    def __init__(self, node):
        self.node = node
        self.model = PinholeCameraModel()
        self.tf_buffer = Buffer()
        TransformListener(self.tf_buffer, self.node)

    def _set_intrinsics(self, camera_info: CameraInfo):
        self.model.fromCameraInfo(camera_info)

    def localize(self,
                 keypoints: dict,
                 depth_image: np.ndarray,
                 camera_info: CameraInfo,
                 camera_frame: str = 'camera',
                 orient_frame: str = 'camera_orientation'
                 ) -> tuple:
        # set camera intrinsics
        self._set_intrinsics(camera_info)

        # get latest transform (orient_frame → camera_frame)
        latest = RosTime()  
        try:
            trans = self.tf_buffer.lookup_transform(
                camera_frame,
                orient_frame,
                latest
            )
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            raise RuntimeError(f"TF lookup failed: {e}")

        # build rotation matrix (camera → leveled frame)
        q = trans.transform.rotation
        quat = [q.x, q.y, q.z, q.w]
        R_cam_to_orient = quaternion_matrix(quaternion_inverse(quat))[:3, :3]

        pts_oriented = []
        for joint in ('left_hip', 'right_hip', 'left_shoulder', 'right_shoulder'):
            if joint not in keypoints:
                continue
            u, v = keypoints[joint]
            ui, vi = int(u), int(v)
            if not (0 <= vi < depth_image.shape[0] and 0 <= ui < depth_image.shape[1]):
                continue

            z = float(depth_image[vi, ui])
            if not np.isfinite(z) or z <= 0:
                continue

            # project pixel to 3D ray and scale by depth
            ray = self.model.projectPixelTo3dRay((u, v))
            pt_cam = np.array(ray) * z

            # rotate into leveled (ground‐parallel) frame
            pts_oriented.append(R_cam_to_orient.dot(pt_cam))

        if not pts_oriented:
            raise ValueError("No valid depth at provided keypoints.")

        # mean X and Z in leveled frame
        p_mean = np.mean(pts_oriented, axis=0)
        return float(p_mean[0]), float(p_mean[2])
