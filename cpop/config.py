import logging
import os

# host and port of local MQTT broker
BROKER_HOST = os.environ.get('CPOP_BROKER_HOST') or 'localhost'
BROKER_PORT = int(os.environ.get('CPOP_BROKER_PORT') or 1883)

# MQTT topic name
MQTT_TOPIC_NAME = 'cpop'

# camera device
CAMERA_WIDTH = int(os.environ.get('CPOP_CAMERA_WIDTH') or 1280)
CAMERA_HEIGHT = int(os.environ.get('CPOP_CAMERA_HEIGHT') or 720)
CAMERA_DEVICE = 0
CAMERA_MODEL = os.environ.get('CPOP_CAMERA_MODEL') or 'default'
FPS = 30

# aruco markers
ARUCO_MARKER_LENGTH = 0.18  # aruco marker side length
ARUCO_MARKER_SET = 'SET_4X4_50'  # which aruco marker set to use (aruco dict)

# configure logging
logging.basicConfig(level=logging.INFO)

# deprecated/legacy
CALIBRATE_FRAME = 'data/calib-and-test/frame_1920x1080.jpg'
