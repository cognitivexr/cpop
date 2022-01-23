import argparse
import time

import logging

from typing import List
from cpop.aruco import detect
from cpop.aruco.detect import ArucoDetector

from cpop import config
from cpop.camera.camera import Camera
from cpop.capture import get_capture_device
from cpop.core import Detection, DetectionStream, ObjectDetector, Point
import cpop.camera.cameradb as cameradb

from cpop.detector.v1.detector import ObjectDetectorV1
from cpop.detector.v2.detector import ObjectDetectorV2

from cpop import config
from cpop.camera.calibrate import run_charuco_detection, run_aruco_detection
from cpop.cli.anchor import get_camera_from_arguments, get_aruco_context_from_args
from cpop.pubsub import DetectionPublishStream, CPOPPublisherMQTT

import numpy as np

import cv2

logger = logging.getLogger(__name__)


class DetectionPrinter(DetectionStream):

    def notify(self, detection: Detection):
        print(detection)


def draw_bounding(bgr, detection: Detection, camera: Camera):
    camera_matrix = camera.intrinsic.camera_matrix
    dist = camera.intrinsic.dist_coeffs

    # TODO:
    p0 = np.array([
        detection.Position.X,
        detection.Position.Y,
        detection.Position.Z
    ])

    delta = (p0/np.linalg.norm(p0))*detection.width
    front = np.array([p1, p2, p3, p4])
    back = np.array([p+delta for p in front])
    axis = np.concatenate((front, back))

    # Now we transform the cube to the marker position and project the resulting points into 2d
    color = (255, 0, 0)
    rvec = np.array([0., 0., 0.]).reshape(-1, 1)
    tvec = rvec
    imgpts, jac = cv2.projectPoints(axis, rvec, tvec, camera_matrix, dist)
    imgpts = np.int32(imgpts).reshape(-1, 2)
    # Now comes the drawing.
    # In this example, I would like to draw the cube so that the walls also get a painted
    # First create a copy of the original picture on which to draw the cube sides (hack for alpha blending)
    sides = bgr.copy()
    # Draw the bottom side (over the marker)
    sides = cv2.drawContours(sides, [imgpts[:4]], -1, color, -2)
    # Draw the top side (opposite of the marker)
    sides = cv2.drawContours(sides, [imgpts[4:]], -1, color, -2)
    # Draw the right side vertical to the marker
    sides = cv2.drawContours(
        sides, [np.array([imgpts[0], imgpts[1], imgpts[5], imgpts[4]])], -1, color, -2)
    # Draw the left side vertical to the marker
    sides = cv2.drawContours(
        sides, [np.array([imgpts[2], imgpts[3], imgpts[7], imgpts[6]])], -1, color, -2)
    # Draw the front side vertical to the marker
    sides = cv2.drawContours(
        sides, [np.array([imgpts[1], imgpts[2], imgpts[6], imgpts[5]])], -1, color, -2)
    # Draw the back side vertical to the marker
    sides = cv2.drawContours(
        sides, [np.array([imgpts[0], imgpts[3], imgpts[7], imgpts[4]])], -1, color, -2)
    # Until here the walls of the cube are drawn in and can be merged
    bgr = cv2.addWeighted(sides, 0.1, bgr, 0.9, 0)
    # Now the edges of the cube are drawn thicker and stronger
    bgr = cv2.drawContours(bgr, [imgpts[:4]], -1, color, 2)
    for ii, j in zip(range(4), range(4, 8)):
        bgr = cv2.line(bgr, tuple(imgpts[ii]), tuple(imgpts[j]), color, 2)
    bgr = cv2.drawContours(bgr, [imgpts[4:]], -1, (255, 0, 0), 2)

    return bgr


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
                        help='display the camera feed in a window')

    parser.add_argument('--realsense', action='store_true',
                        help='use depth sensors of the camera, only works if an intel realsense is connected')

    parser.add_argument('--depth', action='store_true',
                        help='use depth for object positioning')

    args = parser.parse_args()

    camera = get_camera_from_arguments(args)
    cap = camera.get_capture_device(depth=args.depth)

    if args.realsense and args.depth:
        detector = ObjectDetectorV2(camera, ['person'])
    elif args.depth:
        raise Exception('Realsense is not enabled!')
    else:
        detector = ObjectDetectorV1(camera, ['person'])

    try:
        # stream = DetectionPrinter()
        stream = DetectionPublishStream(CPOPPublisherMQTT())
        while True:
            if args.depth:
                more, depth, frame = cap.read()
            else:
                more, frame = cap.read()

            if not more:
                break

            timestamp = time.time()
            if args.depth:
                labels, positions, heights, widths = detector.estimate_object_pose(frame, depth)
            else:
                labels, positions, heights, widths = detector.estimate_object_pose(frame)

            for i in range(len(labels)):
                position = positions[i]
                label = labels[i]
                height = float(heights[i])
                width = float(widths[i])

                detection = Detection(
                    Timestamp=timestamp,
                    Type=label,
                    Position=Point(X=float(position[0]), Y=float(
                        position[1]), Z=float(position[2])),
                    Shape=[Point(X=width, Y=0.0, Z=height)]
                )

                logger.debug('adding detection to stream %s', detection)

                stream.notify(detection)

            if args.show:
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        cap.release()


if __name__ == '__main__':
    main()
