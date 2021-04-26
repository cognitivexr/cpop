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
            cameraMatrix=self.camera.intrinsic.camera_matrix,
            distCoeffs=self.camera.intrinsic.dist_coeffs,
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

    def draw_ar_cubes(self, img_rgb, aruco_poses: ArucoPoseDetections) -> np.array:
        if self.camera is None:
            raise MissingParametersError()

        # , rvecs, tvecs, mtx, dist, marker_size
        rvecs, tvecs = aruco_poses.rvecs, aruco_poses.tvecs
        marker_size = self.context.marker_length
        mtx = self.camera.intrinsic.camera_matrix
        dist = self.camera.intrinsic.dist_coeffs
        # rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(res[0][i], 1, mtx, dist)
        # Define the ar cube
        # Since we previously set a matrix size of 1x1 for the marker and we want the cube to be the same size, it is
        # also defined with a size of 1x1x1
        # It is important to note that the center of the marker corresponds to the origin and we must therefore move 0.5
        # away from the origin
        half_size = marker_size / 2
        axis = np.float32([
            [-half_size, -half_size, 0],
            [-half_size, half_size, 0],
            [half_size, half_size, 0],
            [half_size, -half_size, 0],
            [-half_size, -half_size, marker_size],
            [-half_size, half_size, marker_size],
            [half_size, half_size, marker_size],
            [half_size, -half_size, marker_size]
        ])

        # Now we transform the cube to the marker position and project the resulting points into 2d
        color = (255, 0, 0)
        for i in range(len(rvecs)):
            imgpts, jac = cv2.projectPoints(axis, rvecs[i], tvecs[i], mtx, dist)
            imgpts = np.int32(imgpts).reshape(-1, 2)
            # Now comes the drawing.
            # In this example, I would like to draw the cube so that the walls also get a painted
            # First create six copies of the original picture (for each side of the cube one)
            side1 = img_rgb.copy()
            side2 = img_rgb.copy()
            side3 = img_rgb.copy()
            side4 = img_rgb.copy()
            side5 = img_rgb.copy()
            side6 = img_rgb.copy()
            # Draw the bottom side (over the marker)
            side1 = cv2.drawContours(side1, [imgpts[:4]], -1, color, -2)
            # Draw the top side (opposite of the marker)
            side2 = cv2.drawContours(side2, [imgpts[4:]], -1, color, -2)
            # Draw the right side vertical to the marker
            side3 = cv2.drawContours(side3, [np.array([imgpts[0], imgpts[1], imgpts[5], imgpts[4]])], -1, color, -2)
            # Draw the left side vertical to the marker
            side4 = cv2.drawContours(side4, [np.array([imgpts[2], imgpts[3], imgpts[7], imgpts[6]])], -1, color, -2)
            # Draw the front side vertical to the marker
            side5 = cv2.drawContours(side5, [np.array([imgpts[1], imgpts[2], imgpts[6], imgpts[5]])], -1, color, -2)
            # Draw the back side vertical to the marker
            side6 = cv2.drawContours(side6, [np.array([imgpts[0], imgpts[3], imgpts[7], imgpts[4]])], -1, color, -2)
            # Until here the walls of the cube are drawn in and can be merged
            img_rgb = cv2.addWeighted(side1, 0.1, img_rgb, 0.9, 0)
            img_rgb = cv2.addWeighted(side2, 0.1, img_rgb, 0.9, 0)
            img_rgb = cv2.addWeighted(side3, 0.1, img_rgb, 0.9, 0)
            img_rgb = cv2.addWeighted(side4, 0.1, img_rgb, 0.9, 0)
            img_rgb = cv2.addWeighted(side5, 0.1, img_rgb, 0.9, 0)
            img_rgb = cv2.addWeighted(side6, 0.1, img_rgb, 0.9, 0)
            # Now the edges of the cube are drawn thicker and stronger
            img_rgb = cv2.drawContours(img_rgb, [imgpts[:4]], -1, color, 2)
            for ii, j in zip(range(4), range(4, 8)):
                img_rgb = cv2.line(img_rgb, tuple(imgpts[ii]), tuple(imgpts[j]), color, 2)
            img_rgb = cv2.drawContours(img_rgb, [imgpts[4:]], -1, (255, 0, 0), 2)

        return img_rgb

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
        # return ArucoPoses(detections, rvecs, tvecs, obj_points)
