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

If you have an *Intel Realsense* Camera use the flag `--realsense` to enable the realsense camera.

After calibrating the camera, you can run

    python -m cpop.cli.validate --charuco

and verify that the axis is plotted correctly onto the board. You can also remove the flag `--charuco` to individual
aruco markers, in which case you need to further specify the markers you want to detect. For example,
with `aruco-marker-set SET_4x4_50 --aruco-marker-length 0.18` the script will detect 4x4 markers and calculate camera
positions based on the marker length of 18 cm.
Again use the flag `--realsense` to enable the realsense camera.

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
               rvec: [ 2.00492366 -1.29424414  0.61388495]
               tvec: [-0.30172898 -0.08695201  2.60050308]
    camera position: [-1.72633218 -1.02258803  1.68383734]
    save these values and terminate? (y/n): y

For the values to be correct, it is essential that the `--aruco-marker-length` argument, or the `ARUCO_MARKER_LENGTH`
config parameter is set correctly.
The value describes the length of a marker side in meters.
Add the flag `--realsense` to use a connected realsense camera.

### Run the service

After intrinsic and extrinsic calibration the detector can be started to find the 3D position and bounding box of objects relative to the anchor point.

Make sure a MQTT broker is up and running, check out the `cpop/config.py` for configuration values. Then run

    python -m cpop.cli.detector

which will start sending object positions into the configured MQTT topics.

The `--realsense` flag can be set to make use of Intel Realsense Cameras.
If the `--realsense` and `--depth` flags are set, the object positions and bounding box are calculated with an experimental algorithm using the depth of the Intel Realsense Camera.
Otherwise it is assumed that objects stand on the same plane as the anchor AruCo pattern.

## Build Docker image

### Build container

This container is not GPU accelerated and will likely be pretty slow.
But can be useful to run the calibration scripts.

    docker build -f Dockerfile -t cognitivexr/cpop .

### Build NVIDIA GPU-accelerated container

This requires the [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)

    docker build -f Dockerfile.cuda11 -t cognitivexr/cpop:cuda-11 .

## Demo application

For the DISCE'21 workshop at IEEE VR we developed a simple demo "AR X-Ray Vision" application using Unity.

[![Demo video on YouTube](https://img.youtube.com/vi/nY3PLUTVSbw/0.jpg)](https://www.youtube.com/watch?v=nY3PLUTVSbw)
