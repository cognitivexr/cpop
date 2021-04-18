from typing import NamedTuple

import numpy as np
from cv2 import aruco

from .context import ArucoContext, CameraParameters


class ArucoDetections(NamedTuple):
    corners: np.ndarray
    ids: np.ndarray
    rejected: np.ndarray


class ArucoPoses(NamedTuple):
    detections: ArucoDetections
    rvecs: np.ndarray
    tvecs: np.ndarray
    obj_points: np.ndarray


class ArucoDetector:
    context: ArucoContext
    camera: CameraParameters

    def __init__(self, context: ArucoContext, camera: CameraParameters = None) -> None:
        super().__init__()
        self.context = context
        self.camera = camera

        self._aruco_dict = self.context.marker_set.get_aruco_dictionary()

    def detect_markers(self, frame) -> ArucoDetections:
        corners, ids, rejected = aruco.detectMarkers(frame, dictionary=self._aruco_dict)
        return ArucoDetections(corners, ids, rejected)

    def estimate_marker_poses(self, detections: ArucoDetections):
        rvecs, tvecs, obj_points = aruco.estimatePoseSingleMarkers(
            corners=detections.corners,
            markerLength=self.context.marker_length,
            cameraMatrix=self.camera.camera_matrix,
            distCoeffs=self.camera.dist_coeffs,
            rvecs=None,
            tvecs=None
        )

        return ArucoPoses(detections, rvecs, tvecs, obj_points)
