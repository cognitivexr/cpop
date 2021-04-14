import logging

import cv2


logger = logging.getLogger(__name__)


def get_capture_device(config):
    logger.info('initializing capture device from source %s', config.CAMERA_DEVICE)

    capture = cv2.VideoCapture(config.CAMERA_DEVICE)

    width = config.CAMERA_WIDTH
    height = config.CAMERA_HEIGHT
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    return capture


def capture_from_vid(source=0, width=1920, height=1080, warmup=10):
    """
    Returns a single frame from the given VideoCapture source.

    Args:
        source: the VideoCapture source
        width: frame width
        height: frame height
        warmup: number of frames to be discarded before capturing

    Returns: a single frame

    """
    if warmup < 0:
        raise ValueError('warmup must be positive, was: %s' % warmup)

    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    try:
        frame = None
        for i in range(warmup + 1):
            ret, frame = cap.read()
            if not ret:
                raise ValueError('no more frames to read from source')

        return frame
    finally:
        cap.release()
