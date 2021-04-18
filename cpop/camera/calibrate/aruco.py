import cv2
from cv2 import aruco

from cpop.camera import Camera


def run_aruco_detection(camera: Camera, aruco_dict=None):
    if aruco_dict is None:
        aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_1000)

    cap = camera.get_capture_device()

    params = camera.parameters
    camera_matrix, dist_coeffs = params.camera_matrix, params.dist_coeffs

    def display(frame, rvecs, tvecs):
        if rvecs is not None and tvecs is not None:
            for i in range(len(rvecs)):
                rvec = rvecs[i]
                tvec = tvecs[i]

                try:
                    frame = aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.12)
                except cv2.error:
                    # print('error drawing axis')
                    pass

        # resize
        proportion = max(frame.shape) / 1000.0
        im = cv2.resize(frame, (int(frame.shape[1] / proportion), int(frame.shape[0] / proportion)))

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
                corners, ids, rejected = aruco.detectMarkers(gray, dictionary=aruco_dict)

                # Outline the aruco markers found in our query image
                img = aruco.drawDetectedMarkers(img, corners, ids)
                img = aruco.drawDetectedMarkers(img, rejected, borderColor=(100, 0, 240))

                rvecs, tvecs, obj_points = aruco.estimatePoseSingleMarkers(
                    corners=corners,
                    markerLength=0.12,
                    # 0.12 meters = 12 cm (the huge marker I printed).
                    # but doesn't seem to matter, detects other marker lengths just fine
                    cameraMatrix=camera_matrix,
                    distCoeffs=dist_coeffs,
                    rvecs=None,
                    tvecs=None
                )

                display(img, rvecs, tvecs)

            except cv2.error as e:
                # print("%05d: bad frame")
                display(img, None, None)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
