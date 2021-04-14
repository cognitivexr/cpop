"""
TODO: clean implementation
"""
import abc
import json
import logging
from socket import gethostname

import paho.mqtt.client as mqtt
import paho.mqtt.subscribe as subscribe

import cpop.config as config
from cpop.core import DetectionStream, Detection, DetectionSerializer

logger = logging.getLogger(__name__)


class CPOPPublisher(abc.ABC):
    """ Publisher that publishes CPOP events to the pub/sub broker """

    @abc.abstractmethod
    def publish_event(self, event):
        raise NotImplementedError

    def close(self):
        pass

    @staticmethod
    def get(impl_type=None):
        subclasses = CPOPPublisher.__subclasses__()
        subclasses = {subclass.name(): subclass for subclass in subclasses}
        if not impl_type and len(subclasses) != 1:
            raise Exception('Multiple CPOPPublisher implemtations found and type not specified')
        subclass = subclasses.get(impl_type) or list(subclasses.values())[0]
        return subclass()


class CPOPPublisherMQTT(CPOPPublisher):
    """ Publisher based on MQTT broker """

    @staticmethod
    def name():
        return 'mqtt'

    def __init__(self, client_id=None, topic=None):
        self.client_id = client_id or 'cpop-service-%s' % gethostname()
        self.client = mqtt.Client(client_id=self.client_id)
        self.client.connect(config.BROKER_HOST, config.BROKER_PORT, keepalive=30)
        self.topic = topic or config.MQTT_TOPIC_NAME

    def publish_event(self, event):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('publishing message to topic %s: %s', self.topic, event)

        self.client.publish(self.topic, event)

    def close(self):
        self.client.disconnect()


class CPOPSubscriberMQTT:

    def __init__(self, callback=None):
        self.callback = callback
        pass

    def listen(self):
        subscribe.callback(self.on_message, config.MQTT_TOPIC_NAME,
                           hostname=config.BROKER_HOST, port=config.BROKER_PORT)

    def on_message(self, client, userdata, message):
        print(message)


class JsonSerializer(DetectionSerializer):

    def serialize(self, detection: Detection) -> bytes:
        return json.dumps(self.to_dict(detection)).encode('UTF-8')

    @staticmethod
    def to_dict(detection: Detection):
        return {
            'Timestamp': detection.Timestamp,
            'Type': detection.Type,
            'Position': detection.Position._asdict() if detection.Position else {},
            'Shape': [x._asdict() for x in detection.Shape]
        }


class DetectionPublishStream(DetectionStream):
    """
    DetectionStream implementation that serializes Detection objects as JSON and publishes them using a CPOPPublisher.
    """
    publisher: CPOPPublisher

    def __init__(self, publisher, serializer=None) -> None:
        super().__init__()
        self.publisher = publisher
        self.serializer = serializer or JsonSerializer()

    def notify(self, detection: Detection):
        data = self.serializer.serialize(detection)
        self.publisher.publish_event(data)

    def close(self):
        self.publisher.close()
