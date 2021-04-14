import json
from unittest import TestCase

from cpop.core import Detection, Point
from cpop.pubsub import DetectionPublishStream, CPOPPublisher, JsonSerializer


class FakeCPOPPublisher(CPOPPublisher):

    def __init__(self):
        self.events = list()

    def publish_event(self, event):
        self.events.append(event)


class TestJsonPublisher(TestCase):
    def test(self):
        d = Detection(
            100.123,
            'test',
            Point(1, 2, 3),
            [Point(3, 4, 5)],
        )

        fake = FakeCPOPPublisher()
        publisher = DetectionPublishStream(fake, JsonSerializer())
        publisher.notify(d)

        self.assertEqual(1, len(fake.events))

        expected = {
            "Timestamp": 100.123,
            "Type": "test",
            "Position": {"X": 1, "Y": 2, "Z": 3},
            "Shape": [{"X": 3, "Y": 4, "Z": 5}]
        }

        actual = json.loads(fake.events[0].decode('UTF-8'))

        self.assertEqual(expected, actual)
