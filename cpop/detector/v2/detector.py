import logging
import random
import sys

import torch
from numpy.linalg import norm

from cpop.aruco.detect import *

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


class ObjectDetectorV2:
    """
    First prototype of our CPOP pose estimation used for the demo at DISCE'21. It uses YOLOv5 for detecting objects,
    a custom camera calibration method, and the assumption that objects are on the ground plane for depth estimation.
    """

    def __init__(self, camera: Camera, object_list: List[str] = None):
        # Model
        # For PIL/cv2/np inputs and NMS
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = self._load_model()
        self.model.to(self.device)
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
        self.camera = camera
        self.camera_matrix = camera.intrinsic.camera_matrix
        self.dist = camera.intrinsic.dist_coeffs
        tvec = camera.extrinsic.tvec
        rvec = camera.extrinsic.rvec
        self.set_camera_pose(tvec, rvec)
        self.object_list = object_list or ['person']

    def _load_model(self):
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        if sys.version_info < (3, 8):
            model.autoshape()
        return model

    def set_camera_pose(self, tvec: np.array, rvec: np.array):
        self.tvec = tvec
        self.rvec = rvec
        self.rot_cam_marker = cv2.Rodrigues(self.rvec)[0]
        self.rot_marker_cam = np.transpose(self.rot_cam_marker)
        self.t_cam_marker = self.tvec
        self.pos_cam_marker = np.matmul(-self.rot_marker_cam, self.t_cam_marker)

    def set_camera_matrix(self, camera_matrx: np.array):
        self.camera_matrix = camera_matrx

    def convert_from_uvd(self, u, v, d):
        # d *= self.pxToMetre
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        focalx = self.camera_matrix[0, 0]
        focaly = self.camera_matrix[1, 1]

        x_over_z = (cx - u) / focalx
        y_over_z = (cy - v) / focaly

        z = d / np.sqrt(1. + x_over_z ** 2 + y_over_z ** 2)

        x = x_over_z * z * -1
        y = y_over_z * z * -1

        return np.array([x, y, z])

    def estimate_object_pose(self, frame, depth):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model(frame_rgb, 320 + 32 * 4)  # includes NMS

        positions = []
        heights = []
        widths = []
        labels = []

        points = results.xyxy[0].cpu().numpy()
        for x in points:
            label = self.names[int(x[5])]
            if label in self.object_list:
                depth_val = depth[int((x[1] + x[3]) / 2), int((x[0] + x[2]) / 2)]+0.5
                draw_point(frame, np.array((int((x[0] + x[2]) / 2), int((x[1] + x[3]) / 2))), (0, 255, 255))
                if depth_val == 0:
                    continue
                labels.append(label)

                y1, y2 = int(x[1]), int(x[3])
                x1, x2 = int(x[0]), int(x[2])

                left_top = self.convert_from_uvd(x1, y1, depth_val)
                right_top = self.convert_from_uvd(x2, y1, depth_val)
                right_bot = self.convert_from_uvd(x2, y2, depth_val)
                left_bot = self.convert_from_uvd(x1, y2, depth_val)

                p0 = (left_bot + right_bot) / 2
                print(self.tvec)
                print(p0)

                width = np.linalg.norm(left_top - right_top)
                height = np.linalg.norm(left_top - left_bot)

                positions.append(p0-self.tvec)
                heights.append(height)
                widths.append(width)

        return labels, positions, heights, widths

    def draw_bounding(self, bgr, axis):
        # Now we transform the cube to the marker position and project the resulting points into 2d
        color = (255, 0, 0)
        rvec = np.array([0., 0., 0.]).reshape(-1, 1)
        tvec = rvec
        imgpts, jac = cv2.projectPoints(axis, rvec, tvec, self.camera_matrix, self.dist)
        imgpts = np.int32(imgpts).reshape(-1, 2)
        # Now comes the drawing.
        # In this example, I would like to draw the cube so that the walls also get a painted
        # First create a copy of the original picture on which to draw the cube sides (hack for alpha blending)
        sides = bgr.copy()
        # Draw the bottom side (over the marker)
        sides = cv2.drawContours(sides, [imgpts[:4]], -1, color, -2)
        # Draw the top side (opposite of the marker)
        sides = cv2.drawContours(sides, [imgpts[4:]], -1, color, -2)
        # Draw the right side vertical to the marker
        sides = cv2.drawContours(
            sides, [np.array([imgpts[0], imgpts[1], imgpts[5], imgpts[4]])], -1, color, -2)
        # Draw the left side vertical to the marker
        sides = cv2.drawContours(
            sides, [np.array([imgpts[2], imgpts[3], imgpts[7], imgpts[6]])], -1, color, -2)
        # Draw the front side vertical to the marker
        sides = cv2.drawContours(
            sides, [np.array([imgpts[1], imgpts[2], imgpts[6], imgpts[5]])], -1, color, -2)
        # Draw the back side vertical to the marker
        sides = cv2.drawContours(
            sides, [np.array([imgpts[0], imgpts[3], imgpts[7], imgpts[4]])], -1, color, -2)
        # Until here the walls of the cube are drawn in and can be merged
        bgr = cv2.addWeighted(sides, 0.1, bgr, 0.9, 0)
        # Now the edges of the cube are drawn thicker and stronger
        bgr = cv2.drawContours(bgr, [imgpts[:4]], -1, color, 2)
        for ii, j in zip(range(4), range(4, 8)):
            bgr = cv2.line(bgr, tuple(imgpts[ii]), tuple(imgpts[j]), color, 2)
        bgr = cv2.drawContours(bgr, [imgpts[4:]], -1, (255, 0, 0), 2)

        return bgr
