import cv2

from cpop import config
from cpop.capture import get_capture_device
from cpop.detector.v1.detector import ObjectDetectorV1


def main():
    object_detector = ObjectDetectorV1()
    capture = get_capture_device(config)

    try:
        while True:
            more, frame = capture.read()

            if not more:
                break

            show_frame = frame
            try:
                show_frame = object_detector.init_camera_parameters(frame, viz=True)
            except Exception as e:
                print('error while calibrating: %s' % e)

            cv2.imshow('calibrate', show_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        capture.release()


if __name__ == '__main__':
    main()
