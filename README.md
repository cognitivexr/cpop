CPOP: Cyber-Physical Object Positioning System
==============================================

CPOP tracks physical and digital objects in a shared coordinate space.

## Build and run

To create a virtual environment and install all dependencies, run

    make venv
    source .venv/bin/activate

### Configure

Default CPOP parameters are set in `cpop/config.py`, but most can be overwritten via command line arguments.

### Calibrate the camera

To use CPOP you first need to calibrate the intrinsic camera parameters, and then anchor the camera to a common origin
(extrinsic parameter calibration).
Camera parameters are stored in and loaded from `~/.cpop/cameras`.

#### Intrinsic camera parameters

You can perform the calibration of the intrinsic camera parameters with the following command.

    python -m cpop.cli.calibrate

The calibration uses a 5x7 ChArUco board with 4x4 markers, a square length of 3.5 cm, and marker length of 2.6 cm. You
can [download an A4 PDF here](https://cognitivexr.at/static/files/calib.io_charuco_297x210_5x7_35_DICT_4X4.pdf).

If have an *Intel Realsense* Camera use the flag `--realsense` to enable the realsense camera. Have a look at [Setup](TODO) to see the setup for the camera.

After calibrating the camera, you can run

    python -m cpop.cli.validate --charuco

and verify that the axis is plotted correctly onto the board. You can also remove the flag `--charuco` to individual
aruco markers, in which case you need to further specify the markers you want to detect. For example,
with `aruco-marker-set SET_4x4_50 --aruco-marker-length 0.18` the script will detect 4x4 markers and calculate camera
positions based on the marker length of 18 cm.

#### Extrinsic camera parameters

To perform the extrinsic camera calibration (i.e., anchoring the camera to a common origin), print an ArUco marker with
id = 0 (can be configured) and place it on the floor visible to the camera. Then run

    python -m cpop.cli.anchor --show

The script starts searching for a marker with the id 0 and, if it finds a single marker, will start collecting frames.
The flag `--show` displays an OpenCV window for visual validation, but can be removed to run in headless mode.
If the position of the marker remains stable for several frames, the script will prompt you to save the calculated
parameters and terminate. The output would look something like this:

    state changed: AnchoringState.SEARCHING
    state changed: AnchoringState.STABILIZING
    state changed: AnchoringState.STABLE
               revc: [ 2.00492366 -1.29424414  0.61388495]
               tevc: [-0.30172898 -0.08695201  2.60050308]
    camera position: [-1.72633218 -1.02258803  1.68383734]
    save these values and terminate? (y/n): y

For the values to be correct, it is essential that the `--aruco-marker-length` argument, or the `ARUCO_MARKER_LENGTH`
config parameter is set correctly.
The value describes the length of a marker side in meters.

### Run the service

Make sure a MQTT broker is up and running, check out the `cpop/config.py` for configuration values. Then run

    python -m cpop.cli.detector

which will start sending object positions into the configured MQTT topics.

## Demo application

For the DISCE'21 workshop at IEEE VR we developed a simple demo "AR X-Ray Vision" application using Unity.

[![Demo video on YouTube](https://img.youtube.com/vi/nY3PLUTVSbw/0.jpg)](https://www.youtube.com/watch?v=nY3PLUTVSbw)
