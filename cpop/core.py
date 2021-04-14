import abc
from typing import List, NamedTuple


class Point(NamedTuple):
    X: float
    Y: float
    Z: float


class Detection(NamedTuple):
    Timestamp: float
    Type: str
    Position: Point
    Shape: List[Point]


class DetectionStream(abc.ABC):
    @abc.abstractmethod
    def notify(self, detection: Detection): ...


class ObjectDetector(abc.ABC):
    @abc.abstractmethod
    def process(self, frame, stream: DetectionStream, *args, **kwargs): ...
