import logging
import os

# host and port of local MQTT broker
BROKER_HOST = os.environ.get('BROKER_HOST') or 'localhost'
BROKER_PORT = int(os.environ.get('BROKER_PORT') or 1883)

# camera device
CALIBRATE_FRAME = 'data/calib-and-test/frame_1920x1080.jpg'
CAMERA_WIDTH = 1024
CAMERA_HEIGHT = 576
CAMERA_DEVICE = 0

# MQTT topic name
MQTT_TOPIC_NAME = 'cpop'

# configure logging
logging.basicConfig(level=logging.INFO)