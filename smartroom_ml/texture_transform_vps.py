import math
import operator
import traceback
from functools import reduce

import cv2
import os
import numpy as np

from skimage.util import random_noise

from smartroom_ml.utils.image_transform import rotate_crop
from smartroom_ml.inference import predict_mask, predict_layout
from smartroom_ml.vanishing_point_transforms import find_perspective_border, create_polygon
from smartroom_ml.shadows import transfer_shadows


FLOOR_IDX = 3
WALL_IDX = 0
RUG_IDX = 28
FURNITURE_IDXS = [7, 10, 15, 19, 23, 24, 30, 31, 33, 35, 36, 37, 39, 41, 44, 47, 50, 51, 56, 57, 62, 64, 65, 67, 70,
                  73, 74, 75, 77, 78, 81, 89, 92, 97, 98, 99, 107, 108, 110, 111, 112, 115, 117, 119, 120, 122, 124,
                  125, 127, 129, 130, 131, 132, 135, 137, 138, 139, 141, 142, 143, 145, 147, RUG_IDX]


def multiply_texture(texture, scale):
    row = np.hstack([texture]*scale)
    return np.vstack([row]*scale)


def change_floor_texture(img: np.ndarray, mask: np.ndarray, vps: list, texture: np.ndarray, texture_angle=0,
                         apply_shadows: bool = True, replace_rugs: bool = False,
                         object_mask: np.ndarray = None) -> np.ndarray:
    """

    Args:
        vps: list of 2 vanishing points
        img: orig img
        mask: seg mask of floor
        texture: new floor texture
        texture_angle: angle of texture rotation
        apply_shadows:
        replace_rug: replace rug with texture
        object_mask: optional mask with objects to replace

    Returns:
        Image with changed floor texture
    """
    # Transform texture
    texture = rotate_crop(texture, texture_angle)
    vp1 = vps[0]
    vp2 = vps[1]
    if replace_rugs:
        mask = mask.copy()
        mask = np.where(mask == RUG_IDX, FLOOR_IDX, mask)
    if object_mask is not None:
        replace_mask = object_mask
    else:
        replace_mask = mask
    border = find_perspective_border(create_polygon(np.array(replace_mask == FLOOR_IDX, dtype=np.uint8)), vp1, vp2, img.shape[:-1][::-1])
    center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), border), [len(border)] * 2))
    border = sorted(border, key=lambda coord: (-135 - math.degrees(math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360, reverse=True)
    matrix = cv2.getPerspectiveTransform(
        np.float32([[0, 0], [texture.shape[1] - 1, 0], [texture.shape[1] - 1, texture.shape[0] - 1],
                    [0, texture.shape[0] - 1]]), np.float32(border))
    warped_texture = cv2.warpPerspective(texture, matrix, (img.shape[1], img.shape[0]), cv2.INTER_NEAREST)
    alpha_mask = np.zeros([*replace_mask.shape, 3], dtype=np.uint8)
    alpha_mask[..., 0] = replace_mask == FLOOR_IDX
    alpha_mask[..., 1] = replace_mask == FLOOR_IDX
    alpha_mask[..., 2] = replace_mask == FLOOR_IDX
    result = img.copy() - img * alpha_mask + (warped_texture * alpha_mask)
    if apply_shadows:
        result = transfer_shadows(source_img=img, target_img=result, mask=mask, mask_target=FLOOR_IDX, blur_kernel=int(25 * max(img.shape)/800))
    return result


def change_wall_color(img: np.ndarray, mask: np.ndarray, color: str = '#FFFFFF', use_noise: bool = True,
                      apply_shadows: bool = True, object_mask: np.ndarray = None) -> np.ndarray:
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

    if object_mask is not None:
        replace_mask = object_mask
    else:
        replace_mask = mask
    alpha_mask = np.zeros([*replace_mask.shape, 3], dtype=np.uint8)
    alpha_mask[..., 0] = replace_mask == WALL_IDX
    alpha_mask[..., 1] = replace_mask == WALL_IDX
    alpha_mask[..., 2] = replace_mask == WALL_IDX
    result = img.copy() - img * alpha_mask + (color_image * alpha_mask)
    if apply_shadows:
        result = transfer_shadows(source_img=img, target_img=result, mask=mask, mask_target=WALL_IDX,
                                  dark_trash_scale=1.1, bright_trash_scale=1.5, max_shadow_darkness=0.4,
                                  blur_kernel=int(12 * max(img.shape)/800))

    return result


def polygons_to_mask(polygons, mask_shape):
    mask = np.full(mask_shape, 255, dtype=np.uint8)
    for indx, polygon in polygons.items():
        cv2.drawContours(mask, [np.array([(int(point['x']*mask_shape[1]), int(point['y']*mask_shape[0])) for point in polygon])],
                         -1, (indx), -1)
    return mask

def change_wall_texture(img: np.ndarray, mask: np.ndarray, layout: np.ndarray, vps: list, texture:  np.ndarray,
                        apply_shadows: bool = True, texture_angle: float = 0,
                        object_mask: np.ndarray = None) -> np.ndarray:
    """

        Args:
            vps: list of 3 vanishing points
            img: orig img
            mask: seg mask of floor
            layout: layout mask of room
            texture: new wall texture
            apply_shadows: bool
            texture_angle: float
        Returns:
            Image with changed wall texture
        """
    texture = rotate_crop(texture, texture_angle)
    vp1 = vps[0]
    vp2 = vps[1]
    vp3 = vps[2]
    if isinstance(layout, dict):
        layout_mask = polygons_to_mask(layout, mask.shape)
    else:
        layout_mask = layout
    walls = [(0, 'frontal'), (1, 'left'), (2, 'right')]
    if object_mask is not None:
        replace_mask = object_mask
    else:
        replace_mask = mask
    result = img.copy()
    for idx, wall in walls:
        wall_mask = np.logical_and(layout_mask == idx, replace_mask == WALL_IDX).astype(np.uint8)
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
    from smartroom_ml.remove_objects import find_objects, remove_object_from_mask, remove_objects_lama
    from smartroom_ml.inference import predict_camera_parameters, predict_lama, predict_neurvps

    h, w, x1, y1, x2, y2 = 1170, 780, -118.47391956827687, 384.3497574239535, 1080.5994814938165, 397.6670855398042
    print(predict_camera_parameters(h, w, [(x1, y1), (x2, y2)]))
    exit()
    for name in os.listdir(os.path.join('neurvps_utils', 'demo')):
        print(name)
        if 'png' in name or '0628' in name:
            continue
        base_name = name.split('.')[0]
        img = cv2.imread(os.path.join('neurvps_utils', 'demo', f'{base_name}.jpg'))
        floor_texture = cv2.imread(os.path.join('UprightNet', 'demo', 'laminate.jpg'))
        wall_texture = cv2.imread(os.path.join('perspective_via_vanishing_points', 'demo', 'wall2.jpg'))
        mask = predict_mask(image=img)
        layout = predict_layout(image=img)
        objects = find_objects(mask, FURNITURE_IDXS)
        print(np.unique(objects))
        result_floor = change_floor_texture(img=img,
                                            mask=mask,
                                            texture=floor_texture, texture_angle=90, replace_rugs=False)
        result = change_wall_texture(img=result_floor, mask=mask, layout=layout, texture=wall_texture, texture_angle=1)
        cv2.imwrite(os.path.join('demo', 'furniture_replacement', f'{base_name}.jpg'), np.array(result, dtype=np.uint8))

        for i, object_idx in enumerate(np.unique(objects)[1:]):
            try:
                object_mask = remove_object_from_mask(mask=mask, object_mask=objects==object_idx, layout=layout, floor_idx=FLOOR_IDX,
                                                      wall_idx=WALL_IDX)
                result_floor = change_floor_texture(img=img,
                                                    mask=mask,
                                                    texture=floor_texture, texture_angle=90, replace_rugs=False,
                                                    object_mask=object_mask)
                result = change_wall_texture(img=result_floor, mask=mask, layout=layout, texture=wall_texture, texture_angle=1,
                                             object_mask=object_mask)
                cv2.imwrite(os.path.join('demo','furniture_replacement',f'{base_name}_{i}.jpg'),  np.array(result, dtype=np.uint8))
            except Exception:
                traceback.print_exc()
                print(i)
            # cv2.imshow('asd', np.array(result, dtype=np.uint8))
            # cv2.waitKey(0)
        try:
            object_mask = remove_object_from_mask(mask=mask, object_mask=objects!=0, layout=layout,
                                                  floor_idx=FLOOR_IDX,
                                                  wall_idx=WALL_IDX)
            result_floor = change_floor_texture(img=img,
                                                mask=mask,
                                                texture=floor_texture, texture_angle=90, replace_rugs=False,
                                                object_mask=object_mask)
            result = change_wall_texture(img=result_floor, mask=mask, layout=layout, texture=wall_texture, texture_angle=1,
                                         object_mask=object_mask)
            cv2.imwrite(os.path.join('demo', 'furniture_replacement', f'{base_name}_full.jpg'),
                        np.array(result, dtype=np.uint8))
        except Exception:
            traceback.print_exc()
            print(base_name)
    exit()
    img = cv2.imread(os.path.join('smartroom_ml', 'demo', 'demo1.webp'))
    # img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    mask = predict_mask(img)
    base_name = 'demo1_laminate_wall1'
    floor_texture = cv2.imread(os.path.join('smartroom_ml', 'UprightNet', 'demo', 'laminate.jpg'))
    floor_texture = multiply_texture(floor_texture, 3)
    wall_texture = cv2.imread(os.path.join('smartroom_ml', 'perspective_via_vanishing_points', 'demo', 'wall2.jpg'))
    wall_texture = multiply_texture(wall_texture, 2)
    result = img
    result = change_floor_texture(img=result,
                                  mask=mask,
                                  texture=floor_texture, texture_angle=90, replace_rugs=False)
    cv2.imwrite(os.path.join('smartroom_ml', 'demo', 'result', f'{base_name}_floor.jpeg'),
                result)
    result = change_wall_texture(img=result, mask=mask, texture=wall_texture, texture_angle=0)
    # result_wall = change_wall_color(img,mask, color='#ffa182')
    cv2.imwrite(os.path.join('smartroom_ml', 'demo', 'result', f'{base_name}_floor_wall.jpeg'),
                result)
    cv2.imwrite(os.path.join('smartroom_ml', 'demo', 'result', f'{base_name}_wall.jpeg'),
                change_wall_texture(img=img, mask=mask, texture=wall_texture, texture_angle=0))
    # cv2.imshow('result', result)
    #
    # cv2.waitKey(0)
    exit()
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
        # result_wall = change_wall_texture(img=img, mask=mask, texture=wall_texture)
        result_floor = change_floor_texture(img=img, mask=mask, texture=floor_texture, texture_angle=0, replace_rugs=True)
        print('next')
        cv2.imshow('result', result_floor)
        cv2.waitKey(0)

        # cv2.imwrite(os.path.join('perspective_via_vanishing_points', 'demo', 'result_wall_texture2', img_name), np.hstack([img, result_wall]))
        # cv2.imwrite(os.path.join('perspective_via_vanishing_points', 'demo', 'result', 'mask.png'),
        #             mask)
        # cv2.imshow('orig', img)
        # cv2.imshow('result', result)
        # cv2.waitKey(0)
