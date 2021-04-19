from typing import NamedTuple, List, Union

import cv2
import numpy as np
from cv2 import aruco

from .context import ArucoContext, CameraParameters


class MissingParametersError(Exception):
    pass


class Pose(NamedTuple):
    rvec: np.ndarray
    tvec: np.ndarray


class Marker:
    marker_id: int
    corners: np.ndarray
    pose: Pose = None

    def __init__(self, marker_id: int, corners: np.ndarray, pose: Pose = None):
        self.marker_id = marker_id
        self.corners = corners
        self.pose = pose

    def get_camera_position(self):
        if self.pose is None:
            raise MissingParametersError('marker pose was not yet detected')

        rmat, _ = cv2.Rodrigues(self.pose.rvec)
        rot_marker_cam = np.transpose(rmat)
        pos_cam_marker = np.matmul(-rot_marker_cam, self.pose.tvec)
        return pos_cam_marker


class ArucoDetections:
    corners: np.ndarray
    ids: np.ndarray
    rejected: np.ndarray

    def __init__(self, corners: np.ndarray, ids: np.ndarray, rejected: np.ndarray):
        self.corners = corners
        self.ids = ids
        self.rejected = rejected

        self._markers = None  # lazy loaded structure in ArucoDetections.markers

    def is_empty(self):
        return self.ids is None or len(self.ids) == 0

    @property
    def markers(self) -> List[Marker]:
        if self._markers is None:
            self._markers = self._assemble_markers()
        return self._markers

    def _assemble_markers(self) -> List[Marker]:
        if self.is_empty():
            return []

        ids = self.ids
        corners = self.corners
        return [Marker(marker_id=int(ids[i][0]), corners=corners[i][0]) for i in range(len(ids))]


class ArucoPoseDetections:
    detections: ArucoDetections
    rvecs: np.ndarray
    tvecs: np.ndarray
    obj_points: np.ndarray

    def __init__(self, detections: ArucoDetections, rvecs: np.ndarray, tvecs: np.ndarray, obj_points: np.ndarray):
        self.detections = detections
        self.rvecs = rvecs
        self.tvecs = tvecs
        self.obj_points = obj_points

        self._markers = None

    def is_empty(self):
        return self.rvecs is None or len(self.rvecs) == 0

    @property
    def markers(self) -> List[Marker]:
        if self._markers is None:
            self._markers = self._assemble_markers()
        return self._markers

    def _assemble_markers(self) -> List[Marker]:
        markers = self.detections.markers

        rvecs = self.rvecs
        tvecs = self.tvecs

        for i in range(len(markers)):
            markers[i].pose = Pose(rvecs[i][0], tvecs[i][0])

        return markers


def draw_detections(frame, detections: Union[ArucoDetections, ArucoPoseDetections], rejected=False):
    if isinstance(detections, ArucoPoseDetections):
        d = detections.detections
    else:
        d = detections

    if detections.is_empty():
        return frame

    # Outline the aruco markers found in our query image
    frame = aruco.drawDetectedMarkers(frame, d.corners, d.ids)
    if rejected:
        frame = aruco.drawDetectedMarkers(frame, d.rejected, borderColor=(100, 0, 240))

    return frame


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

    def detect_marker_poses(self, frame) -> ArucoPoseDetections:
        return self.estimate_poses(self.detect_markers(frame))

    def estimate_poses(self, detections: ArucoDetections) -> ArucoPoseDetections:
        if self.camera is None:
            raise MissingParametersError()

        if detections.is_empty():
            return ArucoPoseDetections(detections, np.array([]), np.array([]), np.array([]))

        rvecs, tvecs, obj_points = aruco.estimatePoseSingleMarkers(
            corners=detections.corners,
            markerLength=self.context.marker_length,
            cameraMatrix=self.camera.camera_matrix,
            distCoeffs=self.camera.dist_coeffs,
            rvecs=None,
            tvecs=None
        )

        return ArucoPoseDetections(detections, rvecs, tvecs, obj_points)

    def draw_markers(self, frame, detections: Union[ArucoDetections, ArucoPoseDetections], rejected=False):
        return draw_detections(frame, detections, rejected)

    def draw_axes(self, frame, detections: ArucoPoseDetections):
        if self.camera is None:
            raise MissingParametersError()

        rvecs = detections.rvecs
        tvecs = detections.tvecs

        if rvecs is None:
            return frame

        for i in range(len(rvecs)):
            rvec = rvecs[i]
            tvec = tvecs[i]

            try:
                frame = aruco.drawAxis(
                    image=frame,
                    cameraMatrix=self.camera.camera_matrix,
                    distCoeffs=self.camera.dist_coeffs,
                    rvec=rvec, tvec=tvec,
                    length=self.context.marker_length
                )
            except cv2.error:
                # print('error drawing axis')
                pass

        return frame

    def draw_axis(self, frame, marker: Marker):
        return self.draw_pose_axis(frame, marker.pose)

    def draw_pose_axis(self, frame, pose: Pose):
        if self.camera is None:
            raise MissingParametersError()

        if pose is None:
            return frame

        return aruco.drawAxis(
            image=frame,
            cameraMatrix=self.camera.camera_matrix,
            distCoeffs=self.camera.dist_coeffs,
            rvec=pose.rvec, tvec=pose.tvec,
            length=self.context.marker_length
        )
