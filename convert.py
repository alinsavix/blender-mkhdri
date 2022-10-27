#!/usr/bin/env python3.10
import numpy as np
from PIL import Image
# import py360convert as p360
from cube2equi import cube2equi
import os

def _open_as_PIL(img_path: str) -> Image.Image:
    assert os.path.exists(img_path), f"{img_path} doesn't exist"
    img = Image.open(img_path)
    assert img is not None
    if img.getbands() == tuple("RGBA"):
        # NOTE: Sometimes images are RGBA
        img = img.convert("RGB")
    return img


def load2numpy(img_path: str, dtype: np.dtype, is_cv2: bool = False) -> np.ndarray:
    assert os.path.exists(img_path), f"{img_path} doesn't exist"
    if is_cv2:
        # FIXME: currently only supports RGB
        img = _open_as_cv2(img_path)
    else:
        img = _open_as_PIL(img_path)
        img = np.asarray(img)

    if len(img.shape) == 2:
        img = img[..., np.newaxis]
    img = np.transpose(img, (2, 0, 1))

    # NOTE: Convert dtypes
    # if uint8, keep 0-255
    # if float, convert to 0.0-1.0
    dist_dtype = np.dtype(dtype)
    if dist_dtype in (np.float32, np.float64):
        img = img / 255.0
    img = img.astype(dist_dtype)

    return img


def get_img(path: os.PathLike, dtype: np.dtype = np.dtype(np.float32)):
    # path = os.path.join(DATA_ROOT, IMG_NAME)
    img = load2numpy(path, dtype=dtype, is_cv2=False)
    return img


def _numpy2PIL(img: np.ndarray) -> Image.Image:
    """Supports RGB and grayscale"""

    # FIXME: need to test fro depth image
    # https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes

    if len(img.shape) == 3:
        if img.shape[0] == 3:
            # RGB
            # move the channel to last dim
            img = np.transpose(img, (1, 2, 0))
        elif img.shape[0] == 1:
            # grayscale
            # FIXME: is this necessary?
            img = img.squeeze(0)

    # convert float to uint8
    if img.dtype in (np.float32, np.float64):
        img *= 255
        img = img.astype(np.uint8)

    assert img.dtype == np.uint8
    return Image.fromarray(img)  # if depth, we need to change this to 'F'


# def save(img: Union[np.ndarray, torch.Tensor], path: str) -> None:
def save(img: np.ndarray, path: str) -> None:
    assert len(img.shape) == 3, f"{img.shape} is not a valid input format"
    if isinstance(img, np.ndarray):
        img = _numpy2PIL(img)
        img.save(path)
    # elif torch.is_tensor(img):
    #     img = _torch2PIL(img)
    #     img.save(path)
    else:
        raise ValueError()




img_F = get_img("temp_img_store_front.png")
img_R = get_img("temp_img_store_right.png")
img_B = get_img("temp_img_store_back.png")
img_L = get_img("temp_img_store_left.png")
img_U = get_img("temp_img_store_top.png")
img_D = get_img("temp_img_store_bottom.png")

cube = {
    'F': img_F,
    'R': img_R,
    'B': img_B,
    'L': img_L,
    'U': img_U,
    'D': img_D,
}

eqr = cube2equi(cubemap=cube, width=3840, height=2160, cube_format='dict')
# print('cube shape: ', cube.shape)
print('eqr shape: ', eqr.shape)
# Image.fromarray(eqr.astype(np.uint8)).save("out.png")
save(eqr, "out.png")

#  cube_dice = np.array(Image.open('assert/demo_cube.png'))

# You can make convertion between supported cubemap format
# cube_h = p360.cube_dice2h(cube_dice)  # the inverse is cube_h2dice
# cube_dict = p360.cube_h2dict(cube_h)  # the inverse is cube_dict2h
# cube_list = p360.cube_h2list(cube_h)  # the inverse is cube_list2h
# print('cube_dice.shape:', cube_dice.shape)
# print('cube_h.shape:', cube_h.shape)
# print('cube_dict.keys():', cube_dict.keys())
# print('cube_dict["F"].shape:', cube_dict["F"].shape)
# print('len(cube_list):', len(cube_list))
# print('cube_list[0].shape:', cube_list[0].shape)
