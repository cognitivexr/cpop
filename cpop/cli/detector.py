import argparse
import time

import logging

from typing import List
from cpop.aruco import detect
from cpop.aruco.detect import ArucoDetector

from cpop import config
from cpop.capture import get_capture_device
from cpop.core import Detection, DetectionStream, ObjectDetector, Point
import cpop.camera.cameradb as cameradb

from cpop.detector.v1.detector import ObjectDetectorV1
from cpop.detector.v2.detector import ObjectDetectorV2

from cpop import config
from cpop.camera.calibrate import run_charuco_detection, run_aruco_detection
from cpop.cli.anchor import get_camera_from_arguments, get_aruco_context_from_args

import cv2

logger = logging.getLogger(__name__)

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
    
    parser.add_argument('--realsense', action='store_true',
                        help='use depth sensors of the camera, only works if an intel realsense is connected')

    parser.add_argument('--depth', action='store_true',
                        help='use depth for object positioning')


    args = parser.parse_args()

    camera = get_camera_from_arguments(args)
    cap = camera.get_capture_device(depth=args.depth)

    if args.realsense and args.depth:
        print('Using ObjectDetectorV2...')
        detector = ObjectDetectorV2(camera, ['cup'])
    elif args.depth:
        raise Exception('Realsense is not enabled!')
    else:
        print('Using ObjectDetectorV1...')
        detector = ObjectDetectorV1(camera, ['cup'])

    try:
        stream = DetectionPrinter()  # TODO: replace with real MQTT publisher once done
        while True:
            if args.depth:
                more, depth, frame = cap.read()
            else:
                more, frame = cap.read()
            
            if not more:
                break

            timestamp = time.time()
            if args.depth:
                frame, labels, positions, heights, widths = detector.estimate_object_pose(frame, depth, True)
            else:
                frame, labels, positions, heights, widths = detector.estimate_object_pose(frame, True)

            for i in range(len(labels)):
                print(positions)
                position = positions[i]
                print(position)
                label = labels[i]
                height = float(heights[i])
                width = float(widths[i])

                detection = Detection(
                    Timestamp=timestamp,
                    Type=label,
                    Position=Point(X=float(position[0]), Y=float(position[1]), Z=float(position[2])),
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
