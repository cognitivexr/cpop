CPOP: Cyber-Physical Object Positioning System
==============================================

CPOP tracks physical and digital objects in a shared coordinate space.

## Build and run

To create a virtual environment and install all dependencies, run

    make venv
    source .venv/bin/activate

### Calibrate the camera

You can perform the camera calibration with the following command. The default arguments are set in `cpop/config.py`.

    python -m cpop.cli.calibrate

The calibration uses a 5x7 ChArUco board with 4x4 markers, a square length of 3.5 cm, and marker length of 2.6 cm.
You can [download an A4 PDF here](https://cognitivexr.at/static/files/calib.io_charuco_297x210_5x7_35_DICT_4X4.pdf). 

After calibrating the camera, you can run

    python -m cpop.cli.validate

and verify that the axis is plotted correctly onto the board.
You can also add the flag `--single-marker 4` to detect 4x4 Aruco markers individually.

### Run the service

Make sure a MQTT broker is up and running, check out the `cpop/config.py` for configuration values.
Then run

    python -m cpop.cli.detector

which will start sending object positions into the configured MQTT topics.

## Demo application

For the DISCE'21 workshop at IEEE VR we developed a simple demo "AR X-Ray Vision" application using Unity.

[![Demo video on YouTube](https://img.youtube.com/vi/nY3PLUTVSbw/0.jpg)](https://www.youtube.com/watch?v=nY3PLUTVSbw)
