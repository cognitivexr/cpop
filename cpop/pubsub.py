import logging
from socket import gethostname

import paho.mqtt.client as mqtt
import paho.mqtt.subscribe as subscribe

import cpop.config as config

LOG = logging.getLogger(__name__)


class CPOPPublisher:
    """ Publisher that publishes CPOP events to the pub/sub broker """

    def publish_event(self, event):
        raise Exception('Not implemented')

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

    def __init__(self):
        self.client_id = 'cpop-service-%s' % gethostname()
        self.client = mqtt.Client(client_id=self.client_id)
        self.client.connect(config.BROKER_HOST, config.BROKER_PORT, keepalive=30)

    def publish_event(self, event):
        LOG.debug('publishing message to topic %s: %s', config.MQTT_TOPIC_NAME, event)
        self.client.publish(config.MQTT_TOPIC_NAME, event)


class CPOPSubscriberMQTT:

    def __init__(self, callback=None):
        self.callback = callback
        pass

    def listen(self):
        subscribe.callback(self.on_message, config.MQTT_TOPIC_NAME,
                           hostname=config.BROKER_HOST, port=config.BROKER_PORT)

    def on_message(self, client, userdata, message):
        print(message)
