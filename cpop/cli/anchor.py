import argparse
import sys

import cv2

from cpop import config
from cpop.aruco.context import ArucoMarkerSet, ArucoContext
from cpop.aruco.detect import ArucoDetector
from cpop.camera import cameradb, Camera
from cpop.camera.calibrate.aruco import AnchoringStateMachine, AnchoringState


def print_calibrate_instructions(args):
    print('run the following command to calibrate the camera:', file=sys.stderr)
    print(f'python -m cpop.cli.calibrate '
          f'--width {args.width} '
          f'--height {args.height} '
          f'--device-id {args.device_id} '
          f'--camera-model {args.camera_model}', file=sys.stderr)


def get_camera_from_arguments(args) -> Camera:
    try:
        camera = cameradb.get_camera(args.camera_model, args.width, args.height)
        camera.device_index = args.device_id
        return camera
    except ValueError as e:
        print('could not load camera for parameters: %s' % e, file=sys.stderr)
        print_calibrate_instructions(args)
        return exit(1)


def get_aruco_context_from_args(args) -> ArucoContext:
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
    parser = argparse.ArgumentParser(description='CPOP tool to anchor the camera to an aruco marker as origin')

    parser.add_argument('--width', type=int, required=False, default=config.CAMERA_WIDTH,
                        help='camera capture mode: width')
    parser.add_argument('--height', type=int, required=False, default=config.CAMERA_HEIGHT,
                        help='camera capture mode: height')
    parser.add_argument('--device-id', required=False, default=config.CAMERA_DEVICE,
                        help='camera device id')
    parser.add_argument('--camera-model', type=str, required=False, default=config.CAMERA_MODEL,
                        help='camera model name (for storage)')
    parser.add_argument('--aruco-marker-length', type=float, required=False, default=config.ARUCO_MARKER_LENGTH,
                        help='length of the aruco marker in meters (required for camera position calculation)')
    parser.add_argument('--aruco-marker-set', type=str, required=False, default=config.ARUCO_MARKER_SET,
                        help='the aruco marker set (e.g., SET_4X4_50 or SET_6X6_1000')
    parser.add_argument('--aruco-origin-id', type=int, required=False, default=0,
                        help='the aruco marker id used as origin')
    parser.add_argument('--show', action='store_true',
                        help='display the camera feed in a window and draw the markers')

    args = parser.parse_args()

    camera = get_camera_from_arguments(args)
    aruco_context = get_aruco_context_from_args(args)
    detector = ArucoDetector(aruco_context, camera.intrinsic)
    capture = camera.get_capture_device()

    origin_id = args.aruco_origin_id

    sm = AnchoringStateMachine(origin_id)

    show = args.show

    try:
        while True:
            more, frame = capture.read()

            if not more:
                break

            poses = detector.detect_marker_poses(frame)
            update = sm.process(poses)

            if update.changed:
                print('state changed: %s' % update.to_state)

            if update.changed and sm.state == AnchoringState.STABLE:
                # FIXME: in principle the pose values of the window should be so similar, that it doesn't matter which
                #  value to use. so we use the latest one here. but it would be more robust to find the median from the
                #  window for example.
                extrinsic = sm.calculate_extrinsic_parameters()
                print('           revc:', extrinsic.rvec)
                print('           tevc:', extrinsic.tvec)
                print('camera position:', extrinsic.camera_position())
                print('save these values and terminate? (y/n): ', end='')
                line = sys.stdin.readline()
                if line.strip() in ['yes', 'y']:
                    camera.extrinsic = extrinsic
                    cameradb.save_camera(camera)
                    break

            if show:
                frame = detector.draw_ar_cubes(frame, poses)
                cv2.imshow('frame', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print('interrupted')
    finally:
        capture.release()


if __name__ == '__main__':
    main()
