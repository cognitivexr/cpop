from enum import Enum

import numpy as np
from cv2 import aruco


class ArucoMarkerSet(Enum):
    SET_4X4_50 = 0
    SET_4X4_100 = 1
    SET_4X4_250 = 2
    SET_4X4_1000 = 3
    SET_5X5_50 = 4
    SET_5X5_100 = 5
    SET_5X5_250 = 6
    SET_5X5_1000 = 7
    SET_6X6_50 = 8
    SET_6X6_100 = 9
    SET_6X6_250 = 10
    SET_6X6_1000 = 11
    SET_7X7_50 = 12
    SET_7X7_100 = 13
    SET_7X7_250 = 14
    SET_7X7_1000 = 15
    SET_ARUCO_ORIGINAL = 16
    SET_APRILTAG_16h5 = 17
    SET_APRILTAG_25h9 = 18
    SET_APRILTAG_36h10 = 19
    SET_APRILTAG_36h11 = 20

    def get_aruco_dictionary(self) -> aruco.Dictionary:
        return aruco.getPredefinedDictionary(self.value)


class CameraParameters:
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray

    def __init__(self, camera_matrix, dist_coeffs):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs


class ArucoContext:
    marker_set: ArucoMarkerSet
    marker_length: float

    def __init__(self, marker_set: ArucoMarkerSet, marker_length: float = None):
        self.marker_set = marker_set
        self.marker_length = marker_length


class CharucoContext(ArucoContext):
    rows: int
    columns: int
    square_length: float

    def __init__(self, marker_set, rows, columns, marker_length, square_length):
        super().__init__(marker_set, marker_length)
        self.rows = rows
        self.columns = columns
        self.square_length = square_length

    def create_board(self):
        return aruco.CharucoBoard(
            size=(self.columns, self.rows),
            squareLength=self.square_length,
            markerLength=self.marker_length,
            dictionary=self.marker_set.get_aruco_dictionary()
        )


DefaultArucoContext = ArucoContext(ArucoMarkerSet.SET_ARUCO_ORIGINAL)
