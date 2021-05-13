from cpop.camera.camera import Camera
import logging
import os
import time

import cv2

from cpop import config
from cpop.capture import capture_from_vid
from cpop.core import ObjectDetector, DetectionStream, Detection, Point, CalibrationError
from .detector import ObjectDetectorV2

logger = logging.getLogger(__name__)


class ObjectDetectorDecoratorV2(ObjectDetector):
    detector: ObjectDetectorV2

    def __init__(self, detector: ObjectDetectorV2):
        self.detector = detector

    def process(self, frame, stream: DetectionStream, *args, **kwargs):
        timestamp = time.time()

        _, labels, positions, heights, widths = self.detector.estimate_object_pose(frame, viz=False)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('object_detection took %.4f s', time.time() - timestamp)

        for i in range(len(labels)):
            position = positions[i]
            label = labels[i]
            height = float(heights[i])
            width = float(widths[i])

            detection = Detection(
                Timestamp=timestamp,
                Type=label,
                Position=Point(X=float(position[0]), Y=float(position[1]), Z=float(position[2])),
                Shape=[Point(X=width, Y=0.0, Z=height)]
            )

            logger.debug('adding detection to stream %s', detection)

            stream.notify(detection)


def create_object_detector(camera: Camera) -> ObjectDetector:
    """
    Returns an ObjetDetector backed by an ObjectDetectorV2. If config.CALIBRATE_FRAME is defined, then it uses that to
    calibrate the intrinsic camera parameters. Otherwise it captures a high-res frame from the camera and tries to
    calibrate the camera using that frame.

    :return: an ObjectDetector
    """

    object_detector = ObjectDetectorV2(camera)

    logger.info('camera parameters (rvec: %s, tvec: %s)', object_detector.rvec, object_detector.tvec)

    return ObjectDetectorDecoratorV2(object_detector)
