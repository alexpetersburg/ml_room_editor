import math
import operator
from functools import reduce

import cv2
import os
import numpy as np
from lu_vp_detect import VPDetection
from scipy.ndimage import gaussian_filter
from skimage.util import random_noise

from UprightNet.utils.image_transform import rotate_crop
from inference import predict_mask, predict_layout
from vanishing_point_transforms import find_perspective_border, create_polygon
from shadows import transfer_shadows


FLOOR_IDX = 3
WALL_IDX = 0
RUG_IDX = 28


def multiply_texture(texture, scale):
    row = np.hstack([texture]*scale)
    return np.vstack([row]*scale)


def change_floor_texture(img: np.ndarray, mask: np.ndarray, texture: np.ndarray, texture_angle=0,
                         apply_shadows: bool = True, replace_rugs: bool = False) -> np.ndarray:
    """

    Args:
        img: orig img
        mask: seg mask of floor
        texture: new floor texture
        texture_angle: angle of texture rotation
        apply_shadows:
        replace_rug: replace rug with texture

    Returns:
        Image with changed floor texture
    """
    # Transform texture
    texture = rotate_crop(texture, texture_angle)
    vpd = VPDetection(length_thresh=60, focal_length=max(img.shape)*1.2)
    _ = vpd.find_vps(img)

    vp1 = vpd.vps_2D[0]
    vp2 = vpd.vps_2D[1]
    if replace_rugs:
        mask = mask.copy()
        mask = np.where(mask == RUG_IDX, FLOOR_IDX, mask)
    border = find_perspective_border(create_polygon(np.array(mask == FLOOR_IDX, dtype=np.uint8)), vp1, vp2, img.shape[:-1][::-1])
    center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), border), [len(border)] * 2))
    border = sorted(border, key=lambda coord: (-135 - math.degrees(math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360, reverse=True)
    matrix = cv2.getPerspectiveTransform(
        np.float32([[0, 0], [texture.shape[1] - 1, 0], [texture.shape[1] - 1, texture.shape[0] - 1],
                    [0, texture.shape[0] - 1]]), np.float32(border))
    warped_texture = cv2.warpPerspective(texture, matrix, (img.shape[1], img.shape[0]), cv2.INTER_LINEAR)
    alpha_mask = np.zeros([*mask.shape, 3], dtype=np.uint8)
    alpha_mask[..., 0] = mask == FLOOR_IDX
    alpha_mask[..., 1] = mask == FLOOR_IDX
    alpha_mask[..., 2] = mask == FLOOR_IDX
    result = img.copy() - img * alpha_mask + (warped_texture * alpha_mask)
    if apply_shadows:
        result = transfer_shadows(source_img=img, target_img=result, mask=mask, mask_target=FLOOR_IDX)

    return result


def change_wall_color(img: np.ndarray, mask: np.ndarray, color: str = '#FFFFFF', use_noise: bool = True,
                      apply_shadows: bool = True) -> np.ndarray:
    """

        Args:
            img: orig img
            mask: seg mask of floor
            color: 16-bit hex string
            use_noise: Use noise in color generation
            apply_shadows:

        Returns:
            Image with changed floor texture
        """
    rgb_color = color[-6:]
    if len(rgb_color) != 6:
        raise ValueError(f"Wrong color {color}, Should be hex format")
    rgb_color = (int(rgb_color[:2], 16), int(rgb_color[2:4], 16), int(rgb_color[4:], 16))
    color_image = np.zeros(img.shape, np.uint8)
    color_image[:, :] = rgb_color[::-1]  # convert to bgr
    if use_noise:
        color_image = np.array(random_noise(color_image, var=1e-15) * 255, dtype=np.uint8)
        color_image = cv2.GaussianBlur(color_image, (3, 3), 0)

    alpha_mask = np.zeros([*mask.shape, 3], dtype=np.uint8)
    alpha_mask[..., 0] = mask == WALL_IDX
    alpha_mask[..., 1] = mask == WALL_IDX
    alpha_mask[..., 2] = mask == WALL_IDX
    result = img.copy() - img * alpha_mask + (color_image * alpha_mask)
    if apply_shadows:
        result = transfer_shadows(source_img=img, target_img=result, mask=mask, mask_target=WALL_IDX,
                                  dark_trash_scale=1.1, bright_trash_scale=1.5, max_shadow_darkness=0.4, blur_kernel=12)

    return result


def change_wall_texture(img: np.ndarray, mask: np.ndarray, texture:  np.ndarray,
                        apply_shadows: bool = True) -> np.ndarray:
    """

        Args:
            img: orig img
            mask: seg mask of floor
            texture: new wall texture
            apply_shadows: bool

        Returns:
            Image with changed wall texture
        """
    vpd = VPDetection(length_thresh=60, focal_length=max(img.shape) * 1.2)
    _ = vpd.find_vps(img)
    vp1 = vpd.vps_2D[0]
    vp2 = vpd.vps_2D[1]
    vp3 = vpd.vps_2D[2]

    layout = predict_layout(img)
    walls = [(0, 'frontal'), (1, 'left'), (2, 'right')]
    result = img.copy()
    for idx, wall in walls:
        wall_mask = gaussian_filter(np.logical_and(layout == idx, mask == WALL_IDX).astype(np.uint8) * 255,
                                    sigma=2) / 255 > 0.9
        if wall_mask.sum() == 0:
            continue
        wall_polygon = create_polygon(np.array(wall_mask, dtype=np.uint8))

        if wall == 'left':
            if vp1[0] > max([point[0] for point in wall_polygon]):
                pt1 = vp1
            else:
                pt1 = vp2
            pt2 = vp3
        elif wall == 'right':
            if vp2[0] < min([point[0] for point in wall_polygon]):
                pt1 = vp2
            else:
                pt1 = vp1
            pt2 = vp3
        else:
            pt1 = vp3
            pt2 = None
        border_sorted = []
        border = find_perspective_border(wall_polygon, pt1, pt2, img.shape[:-1][::-1])
        border_max = sorted(border, key=lambda tup: tup[1])
        border_sorted.extend(sorted(border_max[:2], key=lambda tup: tup[0], reverse=True))
        border_sorted.extend(sorted(border_max[2:], key=lambda tup: tup[0]))
        border = border_sorted

        matrix = cv2.getPerspectiveTransform(
            np.float32([[0, 0], [texture.shape[1] - 1, 0], [texture.shape[1] - 1, texture.shape[0] - 1],
                        [0, texture.shape[0] - 1]]), np.float32(border))
        warped_texture = cv2.warpPerspective(texture, matrix, (img.shape[1], img.shape[0]), cv2.INTER_LINEAR)
        alpha_mask = np.zeros([*wall_mask.shape, 3], dtype=np.uint8)
        alpha_mask[..., 0] = wall_mask
        alpha_mask[..., 1] = wall_mask
        alpha_mask[..., 2] = wall_mask
        result = result - result * alpha_mask + (warped_texture * alpha_mask)
    if apply_shadows:
        result = transfer_shadows(source_img=img, target_img=result, mask=mask, mask_target=WALL_IDX,
                                  dark_trash_scale=1.3, bright_trash_scale=1.5, max_shadow_darkness=0.4, blur_kernel=12)
    return result


if __name__ == "__main__":

    for img_name in os.listdir(os.path.join('UprightNet', 'demo', 'input_imgs')):
        if 'mask' in img_name:
            continue
        demo_img_path = os.path.join('UprightNet', 'demo', 'input_imgs', img_name)
        img = cv2.imread(demo_img_path)
        full_hd_scale = min(1920/img.shape[1], 1080/img.shape[0])
        img = cv2.resize(img, (int(img.shape[1]*full_hd_scale), int(img.shape[0]*full_hd_scale)))
        floor_texture = cv2.imread(os.path.join('UprightNet', 'demo', 'wood.jpg'))
        wall_texture = cv2.imread(os.path.join('perspective_via_vanishing_points', 'demo', 'wall1.jpg'))
        mask = predict_mask(image=img)
        result_wall = change_wall_texture(img=img, mask=mask, texture=wall_texture)
        # result_floor = change_floor_texture(img=img, mask=mask, texture=texture, texture_angle=0)

        # cv2.imshow('result', np.vstack([img, result_wall, result_floor]))
        # cv2.waitKey(0)

        cv2.imwrite(os.path.join('perspective_via_vanishing_points', 'demo', 'result_wall_texture2', img_name), np.hstack([img, result_wall]))
        # cv2.imwrite(os.path.join('perspective_via_vanishing_points', 'demo', 'result', 'mask.png'),
        #             mask)
        # cv2.imshow('orig', img)
        # cv2.imshow('result', result)
        # cv2.waitKey(0)
