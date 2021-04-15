CPOP: Cyber-Physical Object Positioning System
==============================================

CPOP tracks physical and digital objects in a shared coordinate space.

## Build and run

To create a virtual environment and install all dependencies, run

    make venv
    source .venv/bin/activate

### Test the camera calibration

You can test the detection of the calibration markers using:

    python -m cpop.cli.calibrate


### Run the service

Make sure a MQTT broker is up and running, check out the `cpop/config.py` for configuration values.
Then run

    python -m cpop.cli.detector

which will start sending object positions into the configured MQTT topics.

## Demo application

For the DISCE'21 workshop at IEEE VR we developed a simple demo "AR X-Ray Vision" application using Unity.

[![Demo video on YouTube](https://img.youtube.com/vi/nY3PLUTVSbw/0.jpg)](https://www.youtube.com/watch?v=nY3PLUTVSbw)
