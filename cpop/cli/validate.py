import argparse
from cpop.aruco.context import ArucoContext, ArucoMarkerSet

import cv2
from cv2 import aruco

from cpop import config
from cpop.camera import cameradb
from cpop.camera.calibrate import run_charuco_detection, run_aruco_detection


def print_calibrate_instructions(args):
    print('run the following command to calibrate the camera:')
    print(f'python -m cpop.cli.calibrate '
          f'--width {args.width} '
          f'--height {args.height} '
          f'--device-id {args.device_id} '
          f'--camera-model {args.camera_model}')


def main():
    parser = argparse.ArgumentParser(
        description='CPOP tool to visually validate a camera calibration')

    parser.add_argument('--width', type=int, required=False, default=config.CAMERA_WIDTH,
                        help='camera capture mode: width')
    parser.add_argument('--height', type=int, required=False, default=config.CAMERA_HEIGHT,
                        help='camera capture mode: height')
    parser.add_argument('--device-id', required=False, default=config.CAMERA_DEVICE,
                        help='camera device id')
    parser.add_argument('--camera-model', type=str, required=False, default='default',
                        help='camera model name (for storage)')
    parser.add_argument('--single-marker', type=int, required=False, default=None,
                        help='use a single marker instead of a ChArUco board by specifying the marker size (4,5,6,7)')
    parser.add_argument('--marker-size', type=float, required=False, default=config.SIZE,
                        help='size of the aruco marker in meters')
    parser.add_argument('--aruco-dict', type=str,
                        required=False, default=config.ARUCO_DICT)
    args = parser.parse_args()

    try:
        camera = cameradb.get_camera(
            args.camera_model, args.width, args.height)
        camera.device_index = args.device_id
    except ValueError as e:
        print('could not load camera for parameters: %s' % e)
        print_calibrate_instructions(args)
        exit(1)

    try:
        if args.single_marker is None:
            run_charuco_detection(camera)
            return

        n = args.single_marker
        if n not in [4, 5, 6, 7]:
            print('marker size out of range, has to be one of: 4,5,6,7')
            exit(1)

        aruco_context = ArucoContext(
            ArucoMarkerSet[args.aruco_dict], args.marker_size)
        run_aruco_detection(camera, aruco_context)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
