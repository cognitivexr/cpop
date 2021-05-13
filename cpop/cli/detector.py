import argparse

from cpop import config
from cpop.capture import get_capture_device
from cpop.core import Detection, DetectionStream
from cpop.detector.v2.factory import create_object_detector
from cpop.camera.cameradb import * 

from cpop import config
from cpop.camera.calibrate import run_charuco_detection, run_aruco_detection
from cpop.cli.anchor import get_camera_from_arguments, get_aruco_context_from_args


                


class DetectionPrinter(DetectionStream):
    def notify(self, detection: Detection):
        print(detection)


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

    parser.add_argument('--live-calibration', type=str, required=False, default=config.ARUCO_MARKER_SET,
                        help='the aruco marker set (e.g., SET_4X4_50 or SET_6X6_1000')
    parser.add_argument('--aruco-marker-set', type=str, required=False, default=config.ARUCO_MARKER_SET,
                        help='the aruco marker set (e.g., SET_4X4_50 or SET_6X6_1000')

    parser.add_argument('--show', action='store_true',
                        help='display the camera feed in a window and draw the markers')

    args = parser.parse_args()
    # TODO: load camera with cam = cameradb.get_camera
    # TODO: set cam.extrinsic = // find_extrinsic_parameters() using ArUco patterns (estimatePoseSingleMarkers)
    camera = get_camera_from_arguments(args)
    # aruco_context = get_aruco_context_from_args(args)
    # TODO: hand intrinsic and extrinsic parameters to detector
    detector = create_object_detector(camera)

    capture = get_capture_device(config)
    try:
        stream = DetectionPrinter()  # TODO: replace with real MQTT publisher once done
        while True:
            more, frame = capture.read()
            if not more:
                break
            detector.process(frame, stream)
    finally:
        capture.release()


if __name__ == '__main__':
    main()
