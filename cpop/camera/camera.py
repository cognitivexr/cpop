from math import tan, pi
from typing import Optional

import cv2
import numpy as np


class CameraParameters:
    width: int  # number of pixels
    height: int  # number of pixels
    camera_matrix: np.ndarray
    dist_coeffs = Optional[np.ndarray]

    def __init__(self, width, height, camera_matrix: np.ndarray, dist_coeffs: Optional[np.ndarray] = None):
        self.width = width
        self.height = height
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

    @staticmethod
    def from_fov(width, height, hfov, vfov) -> 'CameraParameters':
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

        return CameraParameters(width=width, height=height, camera_matrix=camera_matrix)

    @staticmethod
    def from_focal(width, height, focal_mm, pixel_size_mm) -> 'CameraParameters':
        c_x = width / 2
        c_y = height / 2

        sensor_width_mm = focal_mm * pixel_size_mm
        sensor_height_mm = focal_mm * pixel_size_mm
        f_x = (focal_mm / sensor_width_mm) * width
        f_y = (focal_mm / sensor_height_mm) * height

        camera_matrix = np.array([[f_x, 0, c_x],
                                  [0, f_y, c_y],
                                  [0, 0, 1]])

        return CameraParameters(width=width, height=height, camera_matrix=camera_matrix)


class Camera:
    model: str
    parameters: CameraParameters

    def __init__(self, parameters, model: str = None):
        self.parameters = parameters
        self.model = model

    def get_capture_device(self, device_index=0):
        cap = cv2.VideoCapture(device_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.parameters.width)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.parameters.height)
        return cap
