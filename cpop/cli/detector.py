import argparse
from cpop.detector.v2.detector import ObjectDetectorV2
from typing import List
from cpop.aruco import detect
from cpop.aruco.detect import ArucoDetector

from cpop import config
from cpop.capture import get_capture_device
from cpop.core import Detection, DetectionStream
from cpop.detector.v2.factory import create_object_detector
from cpop.camera.cameradb import *

from cpop import config
from cpop.camera.calibrate import run_charuco_detection, run_aruco_detection
from cpop.cli.anchor import get_camera_from_arguments, get_aruco_context_from_args

import cv2


class DetectionPrinter(DetectionStream):
    def notify(self, detection: Detection):
        print(detection)

class DetectionChain(DetectionPrinter):
    ops: List[DetectionStream]

    def __init__(self, ops: List[DetectionStream]):
        self.ops = ops

    def notify(self, detection: Detection):
        for op in self.ops:
            op.notify(detection)

def main():
    parser = argparse.ArgumentParser(description='CPOP object detector v2')

    parser.add_argument('--width', type=int, required=False, default=config.CAMERA_WIDTH,
                        help='camera capture mode: width')
    parser.add_argument('--height', type=int, required=False, default=config.CAMERA_HEIGHT,
                        help='camera capture mode: height')
    parser.add_argument('--device-id', required=False, default=config.CAMERA_DEVICE,
                        help='camera device id')

    parser.add_argument('--camera-model', type=str, required=False, default=config.CAMERA_MODEL,
                        help='camera model name (for storage)')

    parser.add_argument('--live-calibration', action='store_false',
                        help='try calibrating per frame')
    parser.add_argument('--aruco-marker-length', type=float, required=False, default=config.ARUCO_MARKER_LENGTH,
                        help='length of the aruco marker in meters (required for camera position calculation)')
    parser.add_argument('--aruco-marker-set', type=str, required=False, default=config.ARUCO_MARKER_SET,
                        help='the aruco marker set (e.g., SET_4X4_50 or SET_6X6_1000')
    parser.add_argument('--aruco-origin-id', type=int, required=False, default=0,
                        help='the aruco marker id used as origin')

    parser.add_argument('--show', action='store_true',
                        help='display the camera feed in a window and draw the markers')

    args = parser.parse_args()
    # TODO: load camera with cam = cameradb.get_camera
    # TODO: set cam.extrinsic = // find_extrinsic_parameters() using ArUco patterns (estimatePoseSingleMarkers)
    camera = get_camera_from_arguments(args)
    # aruco_context = get_aruco_context_from_args(args)
    # TODO: hand intrinsic and extrinsic parameters to detector

    cap = camera.get_capture_device()
    #detector = create_object_detector(camera, 'cup')
    detector = ObjectDetectorV2(camera, ['cup'])
    if args.live_calibration:            
        aruco_context = get_aruco_context_from_args(args)
        # TODO: pass CameraParameters?
        aruco_detector = ArucoDetector(aruco_context, camera.intrinsic)
    try:
        stream = DetectionPrinter()  # TODO: replace with real MQTT publisher once done
        while True:
            more, frame = cap.read()
            if not more:
                break
            if args.live_calibration:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                try:
                    # Find aruco markers in the query image
                    aruco_detections = aruco_detector.detect_markers(gray)
                    aruco_poses = aruco_detector.estimate_poses(aruco_detections)

                    # output camera position:
                    for marker in aruco_poses.markers:
                        if marker.marker_id == 0:
                            rvec = marker.pose.rvec
                            tvec = marker.pose.tvec
                            detector.set_camera_pose(tvec, rvec)
                            # print(f'camera position: {marker.get_camera_position()}')
                except cv2.error as e:
                    print(e)
            # detector.process(frame, stream)
            frame, labels, positions, heights, widths = detector.estimate_object_pose(frame, True)
            if args.show:
                if args.live_calibration:
                    # Outline the aruco markers found in our query image
                    frame = aruco_detector.draw_markers(frame, aruco_detections, True)
                    if aruco_poses is not None and not aruco_poses.is_empty():
                        # draw_ar_cubes(frame, rvecs, tvecs, camera_matrix, dist_coeffs, aruco_context.marker_length)
                        frame = aruco_detector.draw_ar_cubes(frame, aruco_poses)
                # frame = detector.visualize(frame)
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        cap.release()


if __name__ == '__main__':
    main()
