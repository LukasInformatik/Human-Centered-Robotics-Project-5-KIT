import numpy as np
if not hasattr(np, 'float'):
    np.float = float

from image_geometry import PinholeCameraModel
from sensor_msgs.msg import CameraInfo

class HumanLocalizer:
    """
    Localize a human in the camera frame using:
      - depth image (32FC1)
      - 2D keypoints
    """

    def __init__(self, node):
        self.node = node
        self.model = PinholeCameraModel()

    def _set_intrinsics(self, camera_info: CameraInfo):
        self.model.fromCameraInfo(camera_info)

    def localize(self,
                 keypoints: dict,
                 depth_image: np.ndarray,
                 camera_info: CameraInfo
                 ) -> tuple:
        # set camera intrinsics
        self._set_intrinsics(camera_info)

        pts_cam = []
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
            pts_cam.append(pt_cam)

        if not pts_cam:
            raise ValueError("No valid depth at provided keypoints.")

        # compute mean position in camera frame (X right, Z forward)
        p_mean = np.mean(pts_cam, axis=0)
        return p_mean
