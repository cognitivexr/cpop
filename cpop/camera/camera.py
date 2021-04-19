from math import tan, pi
from typing import Optional

import cv2
import numpy as np


class IntrinsicCameraParameters:
    width: int  # number of pixels
    height: int  # number of pixels
    camera_matrix: np.ndarray
    dist_coeffs = Optional[np.ndarray]

    def __init__(self, width, height, camera_matrix: np.ndarray, dist_coeffs: Optional[np.ndarray] = None):
        self.width = width
        self.height = height
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

    def __str__(self):
        d = self.__dict__
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()

        return 'CameraParameters(%s)' % d

    @staticmethod
    def from_fov(width, height, hfov, vfov) -> 'IntrinsicCameraParameters':
        """
        Crates camera parameters for the given frame shape from the given horizontal and vertical field of view angles.
        """
        c_x = width / 2
        c_y = height / 2

        f_x = c_x / tan(hfov * 0.5 * pi / 180)
        f_y = c_y / tan(vfov * 0.5 * pi / 180)

        camera_matrix = np.array([[f_x, 0, c_x],
                                  [0, f_y, c_y],
                                  [0, 0, 1]])

        return IntrinsicCameraParameters(width=width, height=height, camera_matrix=camera_matrix)

    @staticmethod
    def from_focal(width, height, focal_mm, pixel_size_mm) -> 'IntrinsicCameraParameters':
        c_x = width / 2
        c_y = height / 2

        sensor_width_mm = focal_mm * pixel_size_mm
        sensor_height_mm = focal_mm * pixel_size_mm
        f_x = (focal_mm / sensor_width_mm) * width
        f_y = (focal_mm / sensor_height_mm) * height

        camera_matrix = np.array([[f_x, 0, c_x],
                                  [0, f_y, c_y],
                                  [0, 0, 1]])

        return IntrinsicCameraParameters(width=width, height=height, camera_matrix=camera_matrix)


class ExtrinsicCameraParameters:
    rvec: np.ndarray
    tvec: np.ndarray


class Camera:
    model: str
    intrinsic: IntrinsicCameraParameters
    extrinsic: ExtrinsicCameraParameters

    def __init__(self, intrinsic, model: str = None):
        self.intrinsic = intrinsic
        self.model = model
        self.device_index = 0

    def get_capture_device(self, device_index=None):
        if device_index is None:
            device_index = self.device_index or 0

        cap = cv2.VideoCapture(device_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.intrinsic.width)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.intrinsic.height)

        return cap
