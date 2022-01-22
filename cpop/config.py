import logging
import os

# host and port of local MQTT broker
BROKER_HOST = os.environ.get('BROKER_HOST') or 'localhost'
BROKER_PORT = int(os.environ.get('BROKER_PORT') or 1883)

# MQTT topic name
MQTT_TOPIC_NAME = 'cpop'

# camera device
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_DEVICE = 0
CAMERA_MODEL = 'default'
FPS = 30

# aruco markers
ARUCO_MARKER_LENGTH = 0.18  # aruco marker side length
ARUCO_MARKER_SET = 'SET_4X4_50'  # which aruco marker set to use (aruco dict)

# configure logging
logging.basicConfig(level=logging.INFO)

# deprecated/legacy
CALIBRATE_FRAME = 'data/calib-and-test/frame_1920x1080.jpg'
