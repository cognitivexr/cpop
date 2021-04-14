import logging
import random
from math import tan, pi

import cv2
import numpy as np
import torch
from numpy.linalg import norm

logger = logging.getLogger(__name__)


###########################
# VISUALIZATION FUNCTIONS #
###########################


def draw_line(frame, p1, p2, color, thickness=3):
    r""" Draws a line on the image
    """
    frame = cv2.line(frame,
                     tuple(p1[:2].astype(int)),
                     tuple(p2[:2].astype(int)),
                     color,
                     thickness)
    return frame


def draw_point(blob_frame, p, color=(0, 255, 0)):
    r""" Draws a point on the image
    """
    cv2.circle(blob_frame, tuple(p[:2].astype(int)),
               3, color, -1)


def draw(frame, imgpts):
    """
    Draws a cube on the image
    param np.array imgpts: 8 image points representing the cube
    """
    # corner = corners[0].ravel()
    imgpts = np.int32(imgpts).reshape(-1, 2)
    # draw ground floor in black
    frame = cv2.drawContours(frame, [imgpts[:4]],
                             -1, (0, 0, 0), -3)
    # draw pillars in blue color
    for i, j in zip(range(4), range(4, 8)):
        frame = cv2.line(frame, tuple(imgpts[i]),
                         tuple(imgpts[j]), (255), 1)
    # draw top layer in red color
    frame = cv2.drawContours(frame, [imgpts[4:]], -1, (0, 0, 255), 1)
    return frame


class ObjectDetectorV1:
    """
    First prototype of our CPOP pose estimation used for the demo at DISCE'21. It uses YOLOv5 for detecting objects,
    a custom camera calibration method, and the assumption that objects are on the ground plane for depth estimation.
    """

    def __init__(self, hfov=70.42, vfov=43.3, pixel_size=None, focal_mm=3.67):
        r""" Initializes CameraCalibration module with sensor information

        Either hfov and vfov must be given or the pixel_size
        and the focal length must be passed to initialize the
        camera_matrix in later steps.

        Parameters
        ----------
        hfov : float
            horizontal field of view
        vfov : float
            vertical field of view
        pixel_size : float
            the pixel size in mm
        focal_mm : float
            focal length in mm
        """
        self.hfov = hfov
        self.vfov = vfov
        self.pixel_size = pixel_size
        self.focal_mm = focal_mm
        # Model
        # For PIL/cv2/np inputs and NMS
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).autoshape()
        self.model.to(self.device)
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

    ############################
    # INITIALIZATION FUNCTIONS #
    ############################

    def get_blob_board(self):
        r""" Initializes the blob board parameters
        """
        self.column_count = 6
        self.row_count = 4
        spacing = 40
        # circle_diameter = 30
        # board_width = (column_count-1)*spacing
        # board_height = (row_count-1)*spacing
        blob_board = np.zeros((self.column_count * self.row_count, 3))
        idx = 0
        for column in range(self.column_count):
            for row in range(self.row_count):
                x = column * spacing
                y = row * spacing
                blob_board[idx] = (x, y, 0)  # TODO check
                idx = idx + 1
        return blob_board

    def get_blob_detector(self):
        r"""Initialize SimpleBlobDetector
        """
        # Setup SimpleBlobDetector parameters.
        blobParams = cv2.SimpleBlobDetector_Params()

        # Change thresholds
        blobParams.minThreshold = 8
        blobParams.maxThreshold = 255

        # Filter by Area.
        blobParams.filterByArea = True
        blobParams.minArea = 10
        blobParams.maxArea = 100

        # Filter by Circularity
        blobParams.filterByCircularity = True
        blobParams.minCircularity = 0.8

        # Filter by Convexity
        blobParams.filterByConvexity = True
        blobParams.minConvexity = 0.9

        # Filter by Inertia
        blobParams.filterByInertia = True
        blobParams.minInertiaRatio = 0.4

        # Create a detector with the parameters
        blobDetector = cv2.SimpleBlobDetector_create(blobParams)
        return blobDetector

    def init_camera_matrix(self, shape):
        r""" Calculates the camera_matrix

        Based on the shape and the sensor parameters
        the camera matrix is calculated as numpy.array as:

        | f_x,   0, c_x |
        | 0  , f_y, c_y |
        | 0  ,   0,   1 |

        Parameters
        ----------
        shape : tuple
            the shape as (height, width)

        Returns
        -------
        camera_matrix
            the calculated camera matrix
        """
        height = shape[0]
        width = shape[1]

        c_x = width / 2
        c_y = height / 2

        if self.hfov and self.vfov:
            # From field of view
            f_x = c_x / tan(self.hfov * 0.5 * pi / 180)
            f_y = c_y / tan(self.vfov * 0.5 * pi / 180)
        elif self.pixel_size:
            # From sensor width and height
            sensor_width_mm = self.focal_mm * self.pixel_size
            sensor_height_mm = self.focal_mm * self.pixel_size
            f_x = (self.focal_mm / sensor_width_mm) * width
            f_y = (self.focal_mm / sensor_height_mm) * height
        self.camera_matrix = np.array([[f_x, 0, c_x],
                                       [0, f_y, c_y],
                                       [0, 0, 1]])
        return self.camera_matrix

    def init_camera_parameters(self, frame, viz=True):
        blob_board = self.get_blob_board()
        blob_detector = self.get_blob_detector()
        blob_frame = None
        self.init_camera_matrix(frame.shape)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints = blob_detector.detect(gray)
        if viz:
            blob_frame = cv2.drawKeypoints(
                frame, keypoints,
                np.array([]), (255, 0, 0),
                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # find board with custom algorithm
        self.filter_keypoints(keypoints)

        # visualize board
        if viz:
            blob_frame = cv2.drawKeypoints(
                blob_frame, keypoints,
                np.array([]), (0, 0, 255),
                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imwrite('res/blobs.png', blob_frame)

        # extract image points from keypoints
        image_points = np.array([
            [keypoints[idx].pt[0], keypoints[idx].pt[1]]
            for idx in range(self.column_count * self.row_count)])

        # sort image points after x and y
        sorted_indexes = np.lexsort((image_points[:, 0], image_points[:, 1]))
        image_points = image_points[sorted_indexes]

        # find matching indices for blob and board corners
        # TODO: check out flags = cv2.SOLVEPNP_IPPE_SQUARE | cv2.SOLVEPNP_IPPE
        # TODO: optimize corner detection
        board_corners = np.array(
            [blob_board[i] for i in [0, 20, 23, 3]])
        image_point_corners = np.array(
            [image_points[i] for i in [6, 0, 17, 23]])

        retval, rvec, tvec = cv2.solvePnP(
            board_corners, image_point_corners,
            self.camera_matrix, None,
            cv2.SOLVEPNP_IPPE)

        if not retval:
            raise Exception('could not solve correspondence')
        if viz:
            axis = np.float32(
                [[0, 0, 0],  # origin
                 [40 * 5, 0, 0],  # right axis
                 [40 * 5, 40 * 3, 0],
                 [0, 40 * 3, 0],  # left axis
                 [0, 0, -40 * 3],
                 [40 * 5, 0, -40 * 3],
                 [40 * 5, 40 * 3, -40 * 3],
                 [0, 40 * 3, -40 * 3]]).reshape(-1, 3)
            projected_points, _ = cv2.projectPoints(
                axis, rvec, tvec, self.camera_matrix, None)
            blob_frame = draw(blob_frame, projected_points)

            draw_point(blob_frame,
                       image_point_corners[0], (255, 0, 255))
            draw_point(blob_frame,
                       image_point_corners[1], (255, 255, 255))
            draw_point(blob_frame,
                       image_point_corners[2], (0, 255, 255))
            draw_point(blob_frame,
                       image_point_corners[3], (0, 0, 0))
            draw_point(blob_frame,
                       projected_points[0][0], (255, 0, 255))
            draw_point(blob_frame,
                       projected_points[1][0], (255, 255, 255))
            draw_point(blob_frame,
                       projected_points[2][0], (0, 255, 255))
            draw_point(blob_frame,
                       projected_points[3][0], (0, 0, 0))
            draw_point(blob_frame,
                       np.mean(image_point_corners, axis=0), (0, 0, 255))
            idx = 0
            for point in image_points:
                font = cv2.FONT_HERSHEY_SIMPLEX
                # print(point)
                cv2.putText(blob_frame, f'{idx}',
                            tuple(point.astype(int)),
                            font, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
                idx = idx + 1
            # cv2.imwrite('res/blobs.png', blob_frame)
        self.rvec = rvec
        self.tvec = tvec
        self.rot_cam_chessboard = cv2.Rodrigues(rvec)[0]
        self.rot_chessboard_cam = np.transpose(self.rot_cam_chessboard)
        self.t_cam_chessboard = tvec
        self.pos_cam_chessboard = np.matmul(-self.rot_chessboard_cam, self.t_cam_chessboard)

        return blob_frame

    ###################
    # BOARD DETECTION #
    ###################

    def filter_keypoints(self, keypoints):
        r""" Discards keypoints until only board keypoints are left

        Iteratively removes keypoints by calculating the center
        position of all remaining keypoints and removing the
        keypoint with the highest distance to the center until
        rows*columns keypoints are left.

        """
        x_coords = [p.pt[0] for p in keypoints]
        y_coords = [p.pt[1] for p in keypoints]

        while len(keypoints) > self.row_count * self.column_count:
            _len = len(keypoints)
            centroid_x = sum(x_coords) / _len
            centroid_y = sum(y_coords) / _len
            index = np.argmax([(x_coords[i] - centroid_x) ** 2 +
                               (y_coords[i] - centroid_y) ** 2
                               for i in range(_len)])
            keypoints.pop(index)
            x_coords.pop(index)
            y_coords.pop(index)

    ########################
    # FIND OBJECT POSITION #
    ########################

    def get_intersection(self, ray_origin, ray_dir, plane, plane_d):
        r""" Returns the 3D intersection point
        """
        plane_dot_ray = plane[0] * ray_dir[0] + \
                        plane[1] * ray_dir[1] + plane[2] * ray_dir[2] + plane_d
        if abs(plane_dot_ray) > 0:
            plane_dot_ray_origin = ray_origin[0] * plane[0] + \
                                   ray_origin[1] * plane[1] + ray_origin[2] * plane[2] + plane_d
            return ray_origin - ray_dir * (plane_dot_ray_origin / plane_dot_ray)

    def to_coordinate_plane(self, image_point):
        r""" maps a image point to a coordinate
        """
        # calculate the 3d direction of the ray in camera coordinate frame
        image_point_norm = cv2.undistortPoints(
            image_point, self.camera_matrix, None)[0][0]
        ray_dir_cam = np.array([image_point_norm[0], image_point_norm[1], 1])

        # compute the 3d direction
        # Map the ray direction vector from camera coordinates
        # to chessboard coordinates
        ray_dir_chessboard = np.matmul(self.rot_chessboard_cam, ray_dir_cam)

        # Find the desired 3d point by computing the intersection between the
        # 3d ray and the chessboard plane with Z=0:
        # Expressed in the coordinate frame of the chessboard, the ray
        # originates from the
        # 3d position of the camera center, i.e. 'pos_cam_chessboard',
        #  and its 3d
        # direction vector is 'ray_dir_chessboard'
        # Any point on this ray can be expressed parametrically using
        # its depth 'd':
        # P(d) = pos_cam_chessboard + d * ray_dir_chessboard
        # To find the intersection between the ray and the plane of the
        # chessboard, we compute the depth 'd' for which the Z coordinate
        # of P(d) is equal to zero
        d_intersection = -self.pos_cam_chessboard[2] / ray_dir_chessboard[2]
        intersection_point = self.pos_cam_chessboard.T[0] + d_intersection[0] * ray_dir_chessboard
        return intersection_point

    def estimate_pose(self, frame, viz=True):
        if viz:
            tl = 2
            tf = 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model(frame_rgb, 320 + 32 * 4)  # includes NMS

        person_coordinates = []
        labels = []
        points = results.xyxy[0].cpu().numpy()
        for x in points:
            label = self.names[int(x[5])]
            if label == 'person':
                a = np.array([x[0], x[1]])
                b = np.array([x[2], x[1]])
                c = np.array([x[2], x[3]])
                d = np.array([x[0], x[3]])
                labels.append(label)
                person_coordinates.append([a, b, c, d])
            if viz:
                c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
                color = self.colors[int(x[5])]
                draw_point(frame, x[2:], color)
                cv2.rectangle(frame, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
                t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
                c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                cv2.rectangle(frame, c1, c2, (0, 0, 0), -
                1, cv2.LINE_AA)  # filled
                cv2.putText(frame, label, (c1[0], c1[1] - 2), 0, tl / 3,
                            [225, 255, 255], thickness=tf,
                            lineType=cv2.LINE_AA)
        person_coordinates = np.array(person_coordinates)
        positions = []
        heights = []
        widths = []
        for person_coordinate in person_coordinates:
            point0 = self.to_coordinate_plane(person_coordinate[3])
            point1 = self.to_coordinate_plane(person_coordinate[2])
            pos = (point0 + point1) / 2
            if viz:
                points, _ = cv2.projectPoints(point0,
                                              self.rvec, self.tvec,
                                              self.camera_matrix, None)
                draw_point(frame, points[0][0], (255, 255, 255))
                points, jac = cv2.projectPoints(point1,
                                                self.rvec, self.tvec,
                                                self.camera_matrix, None)
                draw_point(frame, points[0][0], (255, 255, 255))

            bounding_span = point1 - point0
            up_vector = np.array([0, 0, -1])

            # calculate normal vector of plane
            width = norm(bounding_span)
            bounding_span = bounding_span / width
            plane_normal = np.cross(bounding_span, up_vector)

            # to unit vector
            plane_normal = plane_normal / norm(plane_normal)
            plane_d = -np.dot(plane_normal, point1)

            plane_point = (person_coordinate[0] + person_coordinate[1]) / 2
            plane_norm_dir = cv2.undistortPoints(
                plane_point, self.camera_matrix, None)[0][0]
            ray_dir_cam = np.array([plane_norm_dir[0], plane_norm_dir[1], 1])
            ray_dir_cam = ray_dir_cam / norm(ray_dir_cam)
            ray_dir_chessboard = np.matmul(
                self.rot_chessboard_cam, ray_dir_cam)
            ray_origin = self.pos_cam_chessboard.T[0]
            point1 = self.get_intersection(ray_origin, ray_dir_chessboard, plane_normal, plane_d)

            height = point0[2] - point1[2]

            if viz:
                points, _ = cv2.projectPoints(point1, self.rvec, self.tvec, self.camera_matrix, None)
                draw_point(frame, points[0][0], (0, 255, 255))
                print(f'height={height}, width={width}, pos={pos}')

            positions.append(pos)
            heights.append(height)
            widths.append(width)

        # positions = np.array(positions)
        # heights = np.array(heights)
        return frame, labels, positions, heights, widths
