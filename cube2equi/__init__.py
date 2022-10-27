#!/usr/bin/env python3

from typing import Dict, List, Union

import numpy as np

# from equilib.grid_sample import numpy_grid_sample
# from .grid_sample.numpy.bilinear import interp2d

from .numpy import convert2horizon, run

__all__ = ["Cube2Equi", "cube2equi"]

ArrayLike = np.ndarray
CubeMaps = Union[
    # single/batch 'horizon' or 'dice'
    np.ndarray,

    # single 'list'
    List[np.ndarray],

    # batch 'list'
    List[List[np.ndarray]],

    # single 'dict'
    Dict[str, np.ndarray],

    # batch 'dict'
    List[Dict[str, np.ndarray]],
]


class Cube2Equi(object):
    """
    params:
    - w_out, h_out (int): equirectangular image size
    - cube_format (str): input cube format("dice", "horizon", "dict", "list")
    - mode (str): interpolation mode, defaults to "bilinear"

    inputs:
    - cubemap (np.ndarray, torch.Tensor, dict, list)

    returns:
    - equi (np.ndarray, torch.Tensor)
    """

    def __init__(
        self, height: int, width: int, cube_format: str, mode: str = "bilinear"
    ) -> None:
        self.height = height
        self.width = width
        self.cube_format = cube_format
        self.mode = mode

    def __call__(self, cubemap: CubeMaps, **kwargs) -> ArrayLike:
        return cube2equi(
            cubemap=cubemap,
            cube_format=self.cube_format,
            width=self.width,
            height=self.height,
            mode=self.mode,
            **kwargs,
        )


def cube2equi(
    cubemap: CubeMaps,
    cube_format: str,
    height: int,
    width: int,
    mode: str = "bilinear",
    **kwargs,
) -> ArrayLike:
    """
    params:
    - cubemap
    - cube_format (str): ("dice", "horizon", "dict", "list")
    - height, width (int): output size
    - mode (str): "bilinear"

    return:
    - equi (np.ndarray, torch.Tensor)
    """

    assert width % 8 == 0 and height % 8 == 0

    # if _type == "numpy":
    horizon = convert2horizon(
        cubemap=cubemap, cube_format=cube_format
    )
    out = run(
        horizon=horizon, height=height, width=width, mode=mode, **kwargs
    )

    if out.shape[0] == 1:
        out = out.squeeze(0)

    return out
