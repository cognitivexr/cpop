from cpop import config
from cpop.capture import get_capture_device
from cpop.detector.v1 import create_object_detector
from cpop.pubsub import CPOPPublisherMQTT, JsonPublisher


def main():
    detector = create_object_detector()
    capture = get_capture_device(config)

    try:
        stream = JsonPublisher(CPOPPublisherMQTT())

        while True:
            more, frame = capture.read()

            if not more:
                break

            detector.process(frame, stream)
    finally:
        capture.release()


if __name__ == '__main__':
    main()
