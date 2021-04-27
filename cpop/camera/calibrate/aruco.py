from enum import Enum
from typing import NamedTuple, List

import cv2
import numpy as np

from cpop.aruco.context import ArucoContext, DefaultArucoContext
from cpop.aruco.detect import ArucoDetector, ArucoPoseDetections, Marker
from cpop.camera import Camera
from cpop.camera.camera import ExtrinsicCameraParameters


def run_aruco_detection(camera: Camera, aruco_context: ArucoContext = None):
    if aruco_context is None:
        aruco_context = DefaultArucoContext

    cap = camera.get_capture_device()
    aruco_detector = ArucoDetector(aruco_context, camera.intrinsic)

    def display(frame, aruco_poses: ArucoPoseDetections):
        if aruco_poses is not None and not aruco_poses.is_empty():
            # draw_ar_cubes(frame, rvecs, tvecs, camera_matrix, dist_coeffs, aruco_context.marker_length)
            frame = aruco_detector.draw_ar_cubes(img, aruco_poses)

        # resize
        proportion = max(frame.shape) / 1000.0
        im = cv2.resize(frame, (int(frame.shape[1] / proportion), int(frame.shape[0] / proportion)))
        # show the debug image
        cv2.imshow('aruco', im)

    try:
        # Create the arrays and variables we'll use to store info like corners and IDs from images processed
        while True:
            more, img = cap.read()
            if not more:
                break

            # Grayscale the image
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            try:
                # Find aruco markers in the query image
                aruco_detections = aruco_detector.detect_markers(gray)

                # Outline the aruco markers found in our query image
                img = aruco_detector.draw_markers(img, aruco_detections, True)
                aruco_poses = aruco_detector.estimate_poses(aruco_detections)

                # output camera position:
                for marker in aruco_poses.markers:
                    if marker.marker_id == 0:
                        print(f'camera position: {marker.get_camera_position()}')

                display(img, aruco_poses)
            except cv2.error as e:
                print(e)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


class AnchoringState(Enum):
    UNKNOWN = 0
    SEARCHING = 1
    FOUND_TOO_MANY = 2
    STABILIZING = 3
    STABLE = 4


class AnchoringStateChange(NamedTuple):
    changed: bool
    from_state: AnchoringState
    to_state: AnchoringState

    @staticmethod
    def create(prev, cur) -> 'AnchoringStateChange':
        return AnchoringStateChange(prev != cur, prev, cur)


class Positions:
    data: List
    size: int

    def __init__(self, size) -> None:
        super().__init__()
        self.data = []
        self.size = size

    def append(self, element):
        data = self.data

        data.append(element)
        if len(data) > self.size:
            self.data = data[1:]

    def clear(self):
        self.data.clear()

    @property
    def length(self):
        return len(self.data)

    @property
    def is_full(self):
        return self.length == self.size

    def array(self):
        return np.asarray(self.data)

    def relative_spread(self):
        """
        Calculates the relative spread (max - min) / median, which is a normalized measure of spread.
        """
        window = self.array()

        pmin = np.min(window, axis=0)
        pmax = np.max(window, axis=0)
        median = np.median(window, axis=0)

        return np.abs((pmax - pmin) / median)


class AnchoringStateMachine:
    state: AnchoringState

    def __init__(self, origin_id=0, stabilize_frames=30, stability_th=0.1):
        self.state = AnchoringState.UNKNOWN
        self.origin_id = origin_id
        self.window = Positions(stabilize_frames)
        self.stability_th = stability_th
        self.stable_origin = None

    def process(self, poses: ArucoPoseDetections) -> AnchoringStateChange:
        state = self.state
        origin_id = self.origin_id

        origins = np.where(poses.detections.ids == origin_id)[0]
        window = self.window

        if len(origins) == 0:
            self.state = AnchoringState.SEARCHING
            sc = AnchoringStateChange.create(state, self.state)
            if sc.changed:
                window.clear()
            return sc
        elif len(origins) > 1:
            self.state = AnchoringState.FOUND_TOO_MANY
            sc = AnchoringStateChange.create(state, self.state)
            if sc.changed:
                window.clear()
            return sc

        origin = self.get_origin_marker(poses)
        window.append(origin.get_camera_position())

        if self.is_stable(window):
            self.stable_origin = origin
            self.state = AnchoringState.STABLE
        else:
            self.state = AnchoringState.STABILIZING

        return AnchoringStateChange.create(state, self.state)

    def get_origin_marker(self, poses: ArucoPoseDetections) -> Marker:
        for marker in poses.markers:
            if marker.marker_id == self.origin_id:
                return marker

        raise ValueError('no marker with id %s found' % self.origin_id)

    def is_stable(self, window) -> bool:
        if not window.is_full:
            return False

        # we consider the last X measurements as stable if the relative spread is below a threshold
        xyz = window.relative_spread()
        return np.max(xyz) < self.stability_th

    def calculate_extrinsic_parameters(self) -> ExtrinsicCameraParameters:
        # FIXME: calculate [R t] for project matrix?
        marker = self.stable_origin
        return ExtrinsicCameraParameters(marker.pose.rvec, marker.pose.tvec)
