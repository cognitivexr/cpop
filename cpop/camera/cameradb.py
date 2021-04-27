"""
The CameraDB creates and stores Camera and CameraParameter objects.
"""
import json
import logging
import os
from typing import Union, Optional

import numpy as np

from cpop.camera.camera import IntrinsicCameraParameters, Camera, ExtrinsicCameraParameters

logger = logging.getLogger(__name__)

store_dir = os.path.join(os.path.expanduser("~"), '.cpop/cameras/')

known_models = {
    'HD Pro Webcam C920 fov': lambda width, height: IntrinsicCameraParameters.from_fov(width, height, 70.42, 43.3),
}


def get_intrinsic_parameter_file_path(model: str, width: int, height: int) -> str:
    fname = f'{model}__intrinsic_{width}x{height}.json'.lower().replace(' ', '_')
    fpath = os.path.join(store_dir, fname)
    return fpath


def get_extrinsic_parameter_file_path(model: str) -> str:
    fname = f'{model}__extrinsic.json'.lower().replace(' ', '_')
    fpath = os.path.join(store_dir, fname)
    return fpath


def _mkdirp(dir_path: str):
    if os.path.isdir(dir_path):
        return
    if os.path.exists(dir_path):
        raise ValueError(f'path {dir_path} exists but is not a directory')

    logger.debug('creating directory %s', dir_path)
    return os.makedirs(dir_path, exist_ok=True)


def param_to_json(params: Union[IntrinsicCameraParameters, ExtrinsicCameraParameters]) -> str:
    doc = params.__dict__

    for k, v in doc.items():
        if isinstance(v, np.ndarray):
            doc[k] = v.tolist()

    return json.dumps(doc)


def parse_intrinsic_parameters(json_str: str) -> IntrinsicCameraParameters:
    def get_array(dictionary, key):
        arr = dictionary.get(key)
        if arr is None:
            return None
        return np.array(arr)

    doc = json.loads(json_str)

    return IntrinsicCameraParameters(
        width=doc['width'],
        height=doc['height'],
        camera_matrix=get_array(doc, 'camera_matrix'),
        dist_coeffs=get_array(doc, 'dist_coeffs')
    )


def parse_extrinsic_parameters(json_str: str) -> ExtrinsicCameraParameters:
    doc = json.loads(json_str)

    return ExtrinsicCameraParameters(
        rvec=np.array(doc['rvec']),
        tvec=np.array(doc['tvec']),
    )


def load_intrinsic_parameters(fpath) -> IntrinsicCameraParameters:
    with open(fpath, 'r') as fd:
        return parse_intrinsic_parameters(fd.read())


def load_extrinsic_parameters(fpath) -> ExtrinsicCameraParameters:
    with open(fpath, 'r') as fd:
        return parse_extrinsic_parameters(fd.read())


def save_parameters(fpath: str, parameters: Union[IntrinsicCameraParameters, ExtrinsicCameraParameters]):
    _mkdirp(os.path.dirname(fpath))

    with open(fpath, 'w') as fd:
        logger.debug('storing camera parameters into %s', fpath)
        fd.write(param_to_json(parameters))


def save_camera(camera: Camera) -> str:
    if not camera.model:
        raise ValueError('cannot store camera without model name')

    fpath = get_intrinsic_parameter_file_path(camera.model, camera.intrinsic.width, camera.intrinsic.height)
    save_parameters(fpath, camera.intrinsic)

    if camera.extrinsic is not None:
        fpath = get_extrinsic_parameter_file_path(camera.model)
        save_parameters(fpath, camera.extrinsic)

    return fpath


def require_intrinsic_parameters(model: str, width: int, height: int) -> IntrinsicCameraParameters:
    fpath = get_intrinsic_parameter_file_path(model, width, height)

    logger.debug('checking if intrinsic parameter file exists %s', fpath)
    if os.path.isfile(fpath):
        # stored parameter file exists, load it
        logger.debug('loading intrinsic parameters from %s', fpath)
        return load_intrinsic_parameters(fpath)

    if model not in known_models:
        raise ValueError('unknown camera model "%s". available: %s' % (model, ','.join(known_models.keys())))

    param_options = known_models[model]

    if callable(param_options):
        # param_options is a lambda that creates CameraParameters
        parameters = param_options(width, height)
        return parameters
    elif isinstance(param_options, dict):
        parameters = param_options.get((width, height))
        if parameters is None:
            available = ','.join(param_options.keys())
            raise ValueError(f'no parameters for camera "{model}" in mode {width}x{height}. available: {available}')
        return parameters
    else:
        raise TypeError('invalid entry in known_models. this is a build error')


def try_load_extrinsic_parameters(model: str) -> Optional[ExtrinsicCameraParameters]:
    fpath = get_extrinsic_parameter_file_path(model)

    logger.debug('checking if extrinsic parameter file exists %s', fpath)
    if os.path.isfile(fpath):
        # stored parameter file exists, load it
        logger.debug('loading extrinsic parameters from %s', fpath)
        return load_extrinsic_parameters(fpath)

    return None


def get_camera(model: str, width: int, height: int) -> Camera:
    intrinsic = require_intrinsic_parameters(model, width, height)
    extrinsic = try_load_extrinsic_parameters(model)
    return Camera(model=model, intrinsic=intrinsic, extrinsic=extrinsic)
