from cpop.aruco.context import ArucoContext
import cv2
from cv2 import aruco
from cpop.camera import Camera
import numpy as np
from cpop.aruco.detect import ArucoDetector, ArucoPoseDetections


def run_aruco_detection(camera: Camera, aruco_context: ArucoContext):
    # if aruco_dict is None:
    #     aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)

    cap = camera.get_capture_device()
    aruco_detector = ArucoDetector(aruco_context, camera)

    def display(frame, aruco_poses: ArucoPoseDetections):
        if aruco_poses is not None:
            # draw_ar_cubes(frame, rvecs, tvecs, camera_matrix, dist_coeffs, aruco_context.marker_length)
            frame = aruco_detector.draw_ar_cubes(img, aruco_poses)

        # resize
        proportion = max(frame.shape) / 1000.0
        im = cv2.resize(
            frame, (int(frame.shape[1] / proportion), int(frame.shape[0] / proportion)))
        # show the debug image
        cv2.imshow('aruco', im)


    try:
        # Create the arrays and variables we'll use to store info like corners and IDs from images processed
        while True:
            more, img = cap.read()
            if not more:
                break

            # Grayscale the image
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            try:
                # Find aruco markers in the query image
                aruco_detections = aruco_detector.detect_markers(gray)

                # Outline the aruco markers found in our query image
                img = aruco_detector.draw_markers(img, aruco_detections, True)
                aruco_poses = aruco_detector.estimate_poses(aruco_detections)

                # output camera position:
                for marker in aruco_poses.markers:
                    if marker.marker_id==12:
                        print(f'camera position: {marker.get_camera_position()}')

                display(img, aruco_poses)
            except cv2.error as e:
                print(e)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
