import numpy as np
from image_geometry import PinholeCameraModel
from sensor_msgs.msg import CameraInfo

ROBOT_T_CAM = np.array([[0,0,1,0],[-1,0,0,0],[0,-1,0,0],[0,0,0,1]])

class HumanLocalizer:
    """
    Localizes a human in the ground plane relative to the camera using:
      - depth data (32FC1)
      - 2D keypoints
      - IMU orientation (quaternion) to compensate for camera tilt.
    """

    def __init__(self):
        self.model = PinholeCameraModel()

    def _set_intrinsics(self, camera_info: CameraInfo):
        """
        Sets the camera intrinsics from a ROS CameraInfo message.
        """
        self.model.fromCameraInfo(camera_info)

    def localize(self,
                 keypoints: dict,
                 depth_image: np.ndarray,
                 camera_info: CameraInfo,
                 R_cam: np.array 
                 ) -> tuple:
        """
        Returns (x, y) in meters in the leveled ground plane.
        """
        # 1) set camera model from CameraInfo
        self._set_intrinsics(camera_info)

        # 2) collect valid 3D keypoints transformed into world leveled frame
        pts_world = []
        for joint in ['left_hip', 'right_hip', 'left_shoulder', 'right_shoulder']:
            if joint not in keypoints:
                continue
            u, v = keypoints[joint]
            u_i, v_i = int(u), int(v)
            if not (0 <= v_i < depth_image.shape[0] and 0 <= u_i < depth_image.shape[1]):
                continue
            z = float(depth_image[v_i, u_i])
            if not np.isfinite(z) or z <= 0:
                continue

            ray = self.model.projectPixelTo3dRay((u, v))  # unit vector
            pt_cam = np.array(ray) * z                   # point in camera frame
            pt_world = R_cam @ pt_cam                    # rotate to world-leveled frame
            pts_world.append(pt_world)

        if not pts_world:
            raise ValueError("No valid depth at provided keypoints.")

        # 3) mean in leveled/world frame
        p_mean = np.mean(pts_world, axis=0)

        # 4) return horizontal X and forward Z
        return float(p_mean[0]), float(p_mean[2])
