import cv2

from cpop.aruco.context import ArucoContext, ArucoMarkerSet, CameraParameters
from cpop.aruco.detect import ArucoDetector
from cpop.camera import cameradb


def main():
    try:
        cam = cameradb.get_camera('default', 640, 480)
    except ValueError as e:
        print('%s' % e)
        print('run `python -m cpop.cli.calibrate` with the camera parameters')
        exit(1)
        return

    detector = ArucoDetector(
        ArucoContext(ArucoMarkerSet.SET_4X4_50, marker_length=0.18),
        CameraParameters(cam.intrinsic.camera_matrix, cam.intrinsic.dist_coeffs)
    )

    cap = cam.get_capture_device()

    try:
        while True:
            more, frame = cap.read()
            if not more:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detections = detector.detect_marker_poses(gray)

            if not detections.is_empty():
                # draw detected markers
                detector.draw_markers(frame, detections, rejected=True)

                for marker in detections.markers:
                    # find the first marker with id = 0 and print it's position/draw the axis
                    if marker.marker_id == 0:
                        pos = marker.get_camera_position()
                        print('marker id with id 0 found, this is the origin: %s' % pos)
                        detector.draw_axis(frame, marker)
                        break

            cv2.imshow('detections', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
