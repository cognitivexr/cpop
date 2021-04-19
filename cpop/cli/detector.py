from cpop import config
from cpop.capture import get_capture_device
from cpop.core import Detection, DetectionStream
from cpop.detector.v1 import create_object_detector


class DetectionPrinter(DetectionStream):
    def notify(self, detection: Detection):
        print(detection)


def main():
    # TODO: load camera with cam = cameradb.get_camera
    # TODO: set cam.extrinsic = // find_extrinsic_parameters() using ArUco patterns (estimatePoseSingleMarkers)
    # TODO: hand intrinsic and extrinsic parameters to detector

    detector = create_object_detector()
    capture = get_capture_device(config)

    try:
        stream = DetectionPrinter()  # TODO: replace with real MQTT publisher once done

        while True:
            more, frame = capture.read()

            if not more:
                break

            detector.process(frame, stream)
    finally:
        capture.release()


if __name__ == '__main__':
    main()
