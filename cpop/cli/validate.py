import argparse

import cv2

from cpop import config
from cpop.camera.calibrate import run_charuco_detection, run_aruco_detection
from cpop.cli.anchor import get_camera_from_arguments, get_aruco_context_from_args


def main():
    parser = argparse.ArgumentParser(description='CPOP tool to visually validate a camera calibration')

    parser.add_argument('--width', type=int, required=False, default=config.CAMERA_WIDTH,
                        help='camera capture mode: width')
    parser.add_argument('--height', type=int, required=False, default=config.CAMERA_HEIGHT,
                        help='camera capture mode: height')
    parser.add_argument('--device-id', required=False, default=config.CAMERA_DEVICE,
                        help='camera device id')
    parser.add_argument('--camera-model', type=str, required=False, default=config.CAMERA_MODEL,
                        help='camera model name (for storage)')
    parser.add_argument('--charuco', action='store_true',
                        help='validate using the default charuco board')
    parser.add_argument('--aruco-marker-length', type=float, required=False, default=config.ARUCO_MARKER_LENGTH,
                        help='length of the aruco marker in meters (required for camera position calculation)')
    parser.add_argument('--aruco-marker-set', type=str, required=False, default=config.ARUCO_MARKER_SET,
                        help='the aruco marker set (e.g., SET_4X4_50 or SET_6X6_1000')

    args = parser.parse_args()

    camera = get_camera_from_arguments(args)

    try:
        if args.charuco:
            run_charuco_detection(camera)  # TODO create charuco context
        else:
            aruco_context = get_aruco_context_from_args(args)
            run_aruco_detection(camera, aruco_context)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
