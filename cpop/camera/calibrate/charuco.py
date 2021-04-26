import cv2
import numpy as np
from cv2 import aruco

from cpop.camera import IntrinsicCameraParameters, Camera


def create_default_board():
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)

    board = aruco.CharucoBoard_create(
        squaresX=7,
        squaresY=5,
        squareLength=0.035,  # 3.5 cm
        markerLength=0.026,  # 2.6 cm
        dictionary=aruco_dict
    )

    return aruco_dict, board


def sample_random(corners_all, ids_all, n):
    if n >= len(corners_all):
        # sample size is larger than available elements
        return corners_all, ids_all

    i = np.arange(0, len(corners_all), 1)
    i = np.random.choice(i, n)
    return np.asarray(corners_all, dtype=object)[i], np.asarray(ids_all, dtype=object)[i]


def detect_charuco(frame, charuco_board, aruco_dict):
    # grayscale the image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find aruco markers in the query image
    corners, ids, _ = aruco.detectMarkers(image=gray, dictionary=aruco_dict)

    try:
        # Get charuco corners and ids from detected aruco markers
        response, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=ids,
            image=gray,
            board=charuco_board
        )

        return charuco_corners, charuco_ids

    except:
        return None, None


def calibrate_camera_charuco(corners, ids, charuco_board, image_size):
    # Now that we've seen all of our images, perform the camera calibration
    # based on the set of points we've discovered
    calibration, camera_matrix, dist_coeffs, rvecs, tvecs = aruco.calibrateCameraCharuco(
        charucoCorners=corners,
        charucoIds=ids,
        board=charuco_board,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None
    )

    return calibration, camera_matrix, dist_coeffs, rvecs, tvecs


def collect_charuco_detections(capture, aruco_dict, charuco_board):
    # Create the arrays and variables we'll use to store info like corners and IDs from images processed
    corners_all = []  # Corners discovered in all images processed
    ids_all = []  # Aruco ids corresponding to corners discovered
    image_size = None  # Determined at runtime

    good = 0

    def show(frame):
        # resize
        proportion = max(frame.shape) / 800.0
        im = cv2.resize(frame, (int(frame.shape[1] / proportion), int(frame.shape[0] / proportion)))

        # add text
        txt = f'good frames so far: {good}'
        im = cv2.putText(im, txt, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                         1, (255, 0, 0), 2, cv2.LINE_AA)

        # show the debug image
        cv2.imshow('calibration', im)

    while True:
        more, img = capture.read()
        if not more:
            break

        # Grayscale the image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find aruco markers in the query image
        corners, ids, _ = aruco.detectMarkers(
            image=gray, dictionary=aruco_dict)

        # Outline the aruco markers found in our query image
        img = aruco.drawDetectedMarkers(image=img, corners=corners)

        try:
            # Get charuco corners and ids from detected aruco markers
            response, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                markerCorners=corners,
                markerIds=ids,
                image=gray,
                board=charuco_board
            )

            # If a Charuco board was found, let's collect image/corner points
            # Requiring at least 20 squares
            if response > 20:
                # Add these corners and ids to our calibration arrays
                corners_all.append(charuco_corners)
                ids_all.append(charuco_ids)

                # Draw the Charuco board we've detected to show our calibrator the board was properly detected
                img = aruco.drawDetectedCornersCharuco(
                    image=img,
                    charucoCorners=charuco_corners,
                    charucoIds=charuco_ids
                )

                # If our image size is unknown, set it now
                if not image_size:
                    image_size = gray.shape[::-1]

                good += 1
                show(img)

            else:
                show(img)

        except cv2.error as e:
            show(img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    return corners_all, ids_all, image_size


def run_charuco_calibration(source=0, width=None, height=None, max_samples=50) -> IntrinsicCameraParameters:
    aruco_dict, charuco_board = create_default_board()

    cap = cv2.VideoCapture(source)
    if width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    more, frame = cap.read()
    if not more:
        cap.release()
        raise ValueError(
            'could not get frames from capture device %s' % source)

    try:
        print('collecting detections...')
        corners_all, ids_all, image_size = collect_charuco_detections(
            cap, aruco_dict, charuco_board)
    finally:
        cv2.destroyWindow('calibration')
        cap.release()

    if len(corners_all) == 0:
        # happens if no corners are detected
        # TODO: find better exception
        raise Exception('no corners detected')

    n = max_samples
    # print('sampling %d detections...' % n)
    corners, ids = sample_random(corners_all, ids_all, n)

    print('running calibration with %d frames of size %s ...' %
          (len(corners), str(image_size)))
    _, camera_matrix, dist_coeffs, rvecs, tvecs = calibrate_camera_charuco(
        corners, ids, charuco_board, image_size)

    return IntrinsicCameraParameters(image_size[0], image_size[1], camera_matrix, dist_coeffs)


def run_charuco_detection(camera: Camera, device_index: any = None):
    aruco_dict, charuco_board = create_default_board()
    camera_matrix, dist_coeff = camera.intrinsic.camera_matrix, camera.intrinsic.dist_coeffs

    cap = camera.get_capture_device(device_index)

    def display(frame):
        # resize
        proportion = max(frame.shape) / 1000.0
        im = cv2.resize(
            frame, (int(frame.shape[1] / proportion), int(frame.shape[0] / proportion)))

        # show the debug image
        cv2.imshow('calibration', im)

    try:
        while True:
            more, frame = cap.read()

            if not more:
                return

            corners, ids = detect_charuco(frame, charuco_board, aruco_dict)

            if not (corners is None):
                ret, p_rvec, p_tvec = aruco.estimatePoseCharucoBoard(
                    charucoCorners=corners,
                    charucoIds=ids,
                    board=charuco_board,
                    cameraMatrix=camera_matrix,
                    distCoeffs=dist_coeff,
                    rvec=None,
                    tvec=None
                )
                try:
                    frame = aruco.drawAxis(
                        frame, camera_matrix, dist_coeff, p_rvec, p_tvec, 0.1)
                except cv2.error as e:
                    pass
            display(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
