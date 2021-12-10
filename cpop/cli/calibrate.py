import argparse

from cpop import config
from cpop.camera import cameradb, Camera
from cpop.camera.calibrate import run_charuco_calibration
from cpop.camera.camera import IntrinsicCameraParameters


def main():
    parser = argparse.ArgumentParser(description='CPOP camera calibration tool to determine intrinsic parameters')

    parser.add_argument('--width', type=int, required=False, default=config.CAMERA_WIDTH,
                        help='camera capture mode: width')
    parser.add_argument('--height', type=int, required=False, default=config.CAMERA_HEIGHT,
                        help='camera capture mode: height')
    parser.add_argument('--fps', type=int, required=False, default=config.FPS,
                        help='camera capture mode: fps')
    parser.add_argument('--device-id', required=False, default=config.CAMERA_DEVICE,
                        help='camera device id')
    parser.add_argument('--camera-model', type=str, required=False, default=config.CAMERA_MODEL,
                        help='camera model name (for storage)')
    parser.add_argument('--max-samples', type=int, required=False, default=50,
                        help='maximum number of frames to use for calibration. higher number will take exponentially '
                             'longer, but may produce more precise results')
    parser.add_argument('--realsense', action='store_true',
                        help='use depth sensors of the camera, only works if an intel realsense is connected')

    args = parser.parse_args()

    print('press q to stop collection and start calibration')
    camera = Camera(model=args.camera_model,
                    intrinsic=IntrinsicCameraParameters(args.width, args.height),
                    realsense=args.realsense,
                    fps=args.fps)

    parameters = run_charuco_calibration(
        camera,
        max_samples=args.max_samples
    )

    camera = Camera(model=args.camera_model, intrinsic=parameters)
    path = cameradb.save_camera(camera)
    print('calibration done! parameters saved to', path)


if __name__ == '__main__':
    main()
