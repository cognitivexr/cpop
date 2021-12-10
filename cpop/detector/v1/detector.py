import logging
import random
from math import tan, pi
from typing import List

import cv2
import numpy as np
import torch
from numpy.linalg import norm

from cpop.camera.camera import Camera

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

    def __init__(self, camera: Camera, object_list: List[str]=['person']):
        self.camera = camera
        self.camera_matrix = camera.intrinsic.camera_matrix
        self.dist = camera.intrinsic.dist_coeffs
        tvec = camera.extrinsic.tvec
        rvec = camera.extrinsic.rvec
        self.set_camera_pose(tvec, rvec)
        self.object_list = object_list
        # Model
        # For PIL/cv2/np inputs and NMS
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).autoshape()
        self.model.to(self.device)
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]


    def set_camera_pose(self, rvec, tvec):
        self.rvec = rvec
        self.tvec = tvec
        self.rot_cam_chessboard = cv2.Rodrigues(rvec)[0]
        self.rot_chessboard_cam = np.transpose(self.rot_cam_chessboard)
        self.t_cam_chessboard = tvec
        self.pos_cam_chessboard = np.matmul(-self.rot_chessboard_cam, self.t_cam_chessboard)

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
            image_point, self.camera_matrix, self.dist)[0][0]
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

    def estimate_object_pose(self, frame, viz=True):
        if viz:
            tl = 2
            tf = 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model(frame_rgb, 320 + 32 * 4)  # includes NMS

        object_coordinates = []
        labels = []
        points = results.xyxy[0].cpu().numpy()
        for x in points:
            label = self.names[int(x[5])]
            if label in self.object_list:
                a = np.array([x[0], x[1]])
                b = np.array([x[2], x[1]])
                c = np.array([x[2], x[3]])
                d = np.array([x[0], x[3]])
                labels.append(label)
                object_coordinates.append([a, b, c, d])
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
        object_coordinates = np.array(object_coordinates)
        positions = []
        heights = []
        widths = []
        for person_coordinate in object_coordinates:
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
