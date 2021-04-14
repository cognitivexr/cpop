import logging
import os
import time

import cv2

from cpop import config
from cpop.capture import capture_from_vid
from cpop.core import ObjectDetector, DetectionStream, Detection, Point, CalibrationError
from .detector import ObjectDetectorV1

logger = logging.getLogger(__name__)


class ObjectDetectorDecorator(ObjectDetector):
    detector: ObjectDetectorV1

    def __init__(self, detector: ObjectDetectorV1):
        self.detector = detector

    def process(self, frame, stream: DetectionStream, *args, **kwargs):
        timestamp = time.time()

        _, labels, positions, heights, widths = self.detector.estimate_pose(frame, viz=False)

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


def create_object_detector() -> ObjectDetector:
    """
    Returns an ObjetDetector backed by an ObjectDetectorV1. If config.CALIBRATE_FRAME is defined, then it uses that to
    calibrate the intrinsic camera parameters. Otherwise it captures a high-res frame from the camera and tries to
    calibrate the camera using that frame.

    :return: an ObjectDetector
    """
    logger.info('initializing camera parameters')
    if config.CALIBRATE_FRAME and os.path.isfile(config.CALIBRATE_FRAME):
        logger.info('using existing frame for calibration %s', config.CALIBRATE_FRAME)
        frame = cv2.imread(config.CALIBRATE_FRAME)
    else:
        logger.info('capturing frame from source')
        frame = capture_from_vid(source=config.CAMERA_DEVICE, width=1920, height=1080)

    if frame is None:
        raise CalibrationError('no frame captured to initialize camera parameters')

    object_detector = ObjectDetectorV1()

    try:
        object_detector.init_camera_parameters(frame, viz=False)
    except IndexError as e:
        raise CalibrationError('error in init_camera_parameters. missing/bad calibration marker?') from e

    logger.info('camera parameters (rvec: %s, tvec: %s)', object_detector.rvec, object_detector.tvec)

    return ObjectDetectorDecorator(object_detector)
