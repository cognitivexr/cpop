import argparse
import sys

import cv2

from cpop import config
from cpop.aruco.context import ArucoContext, ArucoMarkerSet
from cpop.camera import cameradb
from cpop.camera.calibrate import run_charuco_detection, run_aruco_detection


def print_calibrate_instructions(args):
    print('run the following command to calibrate the camera:')
    print(f'python -m cpop.cli.calibrate '
          f'--width {args.width} '
          f'--height {args.height} '
          f'--device-id {args.device_id} '
          f'--camera-model {args.camera_model}')


def aruco_context_from_args(args):
    try:
        marker_set = ArucoMarkerSet[args.aruco_marker_set]
    except KeyError:
        print('unknown marker set "%s"' % args.aruco_marker_set, file=sys.stderr)
        return exit(1)

    marker_len = args.aruco_marker_length
    if marker_len <= 0:
        print('marker length needs to be positive float', file=sys.stderr)
        return exit(0)

    return ArucoContext(marker_set, args.aruco_marker_length)


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

    try:
        camera = cameradb.get_camera(args.camera_model, args.width, args.height)
        camera.device_index = args.device_id
    except ValueError as e:
        print('could not load camera for parameters: %s' % e, file=sys.stderr)
        print_calibrate_instructions(args)
        return exit(1)

    try:
        if args.charuco:
            run_charuco_detection(camera)  # TODO create charuco context
        else:
            aruco_context = aruco_context_from_args(args)
            run_aruco_detection(camera, aruco_context)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
