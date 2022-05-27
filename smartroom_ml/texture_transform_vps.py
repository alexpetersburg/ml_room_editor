import math
import operator
import traceback
from functools import reduce
from typing import Union, Optional

import cv2
import os
import numpy as np
from shapely.geometry import Polygon, Point

from skimage.util import random_noise

from smartroom_ml.utils.image_transform import rotate_crop
from smartroom_ml.inference import predict_mask, predict_layout
from smartroom_ml.vanishing_point_transforms import find_perspective_border, create_polygon, line_intersection, \
                                                    angle_between
from smartroom_ml.shadows import transfer_shadows


FLOOR_IDX = 3
WALL_IDX = 0
RUG_IDX = 28
FURNITURE_IDXS = [7, 10, 15, 19, 23, 24, 30, 31, 33, 35, 36, 37, 39, 41, 44, 47, 50, 51, 56, 57, 62, 64, 65, 67, 70,
                  73, 74, 75, 77, 78, 81, 89, 92, 97, 98, 99, 107, 108, 110, 111, 112, 115, 117, 119, 120, 122, 124,
                  125, 127, 129, 130, 131, 132, 135, 137, 138, 139, 141, 142, 143, 145, 147, RUG_IDX]

LAYOUT_WALL_INDEXES = [0, 1, 2]
LAYOUT_FLOOR_INDEX = 3
GET_LAYOUT_TYPE = {0: 'frontal', 1: 'left', 2: 'right', 3: 'floor', 4: 'celling', 11: 'vp1', 12: 'vp2', 10: 'wall'}


def multiply_texture(texture, scale):
    row = np.hstack([texture]*scale)
    return np.vstack([row]*scale)


def polygons_to_mask(polygons, mask_shape, fill_value: int = 255, polygon_key: None = 'layout_type',
                     default_polygon_value: np.int = 0):
    mask = np.full(mask_shape, fill_value, dtype=np.uint8)
    # [{points: [{'x': 0.00306, 'y': 0.0}, ...]
    #   material: str or np.ndarray
    #   (optional) layout_type: int}]
    for polygon in polygons:
        value = default_polygon_value
        if polygon_key is not None:
            value = polygon.get(polygon_key, default_polygon_value)
        cv2.drawContours(mask, [np.array([(int(point['x']*mask_shape[1]), int(point['y']*mask_shape[0])) for point in polygon['points']])],
                         -1, (value), -1)
    return mask


def determine_vps(vps, layout_type, points):
    vp1 = vps[0]
    vp2 = vps[1]
    vp3 = vps[2]
    if layout_type in ['floor', 'celling']:
        pt1 = vp1
        pt2 = vps[1]
        shapely_polygon = Polygon(points)
        if Point(pt1).within(shapely_polygon) or Point(pt2).within(shapely_polygon):
            pt1 = None
            pt2 = None
    elif layout_type == 'left':
        if vp1[0] > max([point[0] for point in points]):
            pt1 = vp1
        else:
            pt1 = vp2
        pt2 = vp3
    elif layout_type == 'right':
        if vp2[0] < min([point[0] for point in points]):
            pt1 = vp2
        else:
            pt1 = vp1
        pt2 = vp3
    elif layout_type == 'vp1':
        pt1 = vp1
        pt2 = vp3
    elif layout_type == 'vp2':
        pt1 = vp2
        pt2 = vp3
    else:
        pt1 = vp3
        pt2 = None
    return pt1, pt2


def get_polygon_wall_type(points, vps, points_scale=(1, 1)):
    vp1 = np.array(vps[0])
    vp2 = np.array(vps[1])
    y_sorted_points = sorted([(points_scale[0] * point['x'], points_scale[1] * point['y']) for point in points],
                             key=lambda x: x[1])
    if len(points) > 4:

        # start_bot_line_index = None
        # start_top_line_index = None
        # max_top_dist = -1
        # max_bot_dist = -1
        # is_bot_prev_point = points[-1]['y'] < 0.5
        # for i in range(len(points)):
        #     if (points[-1]['y'] < 0.5) == (points[-1]['y'] < 0.5):
        try:
            bot_points = list(filter(lambda x: x[1] < 0.5*points_scale[1], y_sorted_points))
            bot_points_last_index = y_sorted_points.index(bot_points[-1])
            bot_points = y_sorted_points[0: max(2, bot_points_last_index + 1)]
            bot_line = (min(bot_points, key=lambda x: x[0]), max(bot_points, key=lambda x: x[0]))
            top_points = y_sorted_points[(min((len(y_sorted_points) - 2), bot_points_last_index + 1)):]
            top_line = (min(top_points, key=lambda x: x[0]), max(top_points, key=lambda x: x[0]))
        except IndexError:
            bot_line = y_sorted_points[0:2]
            top_line = y_sorted_points[-2:]
    else:

        bot_line = y_sorted_points[0:2]
        top_line = y_sorted_points[-2:]
    intersection_point = np.array(line_intersection(bot_line, top_line, allow_parallel_intersection=True))
    if (np.abs(intersection_point).max() / max(np.abs(vp1).max(), np.abs(vp2).max())) > 10 or \
            (abs(angle_between(np.array(top_line), np.array(((0, 0), (1, 0))))) < 0.174533 and
             abs(angle_between(np.array(bot_line), np.array(((0, 0), (1, 0))))) < 0.174533):
        return 0
    vp1_dist = np.linalg.norm(vp1 - intersection_point)
    vp2_dist = np.linalg.norm(vp2 - intersection_point)
    shapely_polygon = Polygon([(points_scale[0] * point['x'], points_scale[1] * point['y']) for point in points])
    vp1_within_poly = Point(vp1).within(shapely_polygon)
    vp2_within_poly = Point(vp2).within(shapely_polygon)
    if vp1_within_poly and not vp2_within_poly:
        return 12
    if vp2_within_poly and not vp1_within_poly:
        return 11
    if vp2_within_poly and vp1_within_poly:
        return 0

    if vp1_dist <= vp2_dist:
        return 11
    else:
        return 12


def change_floor_texture(img: np.ndarray, mask: np.ndarray, vps: list, texture: np.ndarray, texture_angle=0,
                         apply_shadows: bool = True, replace_rugs: bool = False, object_mask: np.ndarray = None,
                         layout: Union[dict, np.ndarray, None] = None) -> np.ndarray:
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
        layout: mask of floor or polygon {points: [{'x': 0.00306, 'y': 0.0}, ...]}

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
    if layout is not None:
        if isinstance(layout, np.ndarray):
            layout_mask =layout
        else:
            layout_mask = cv2.drawContours(np.zeros(replace_mask.shape),
                                           [np.array([(int(point['x']*mask.shape[1]), int(point['y']*mask.shape[0]))
                                                      for point in layout['points']])], -1, (LAYOUT_FLOOR_INDEX), -1)
        replace_mask = (replace_mask * (layout_mask == LAYOUT_FLOOR_INDEX)).astype(np.uint8)
        mask = mask.copy()
        mask = (mask * (layout_mask == LAYOUT_FLOOR_INDEX))
    border = find_perspective_border(create_polygon(np.array(replace_mask == FLOOR_IDX, dtype=np.uint8)), vp1, vp2, img.shape[:-1][::-1])
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
        result = transfer_shadows(source_img=img, target_img=result, mask=mask, mask_target=FLOOR_IDX,
                                  blur_kernel=int(25 * max(img.shape)/800))
    return result


def _change_polygon_color(img: np.ndarray, alpha_mask: np.ndarray, color: str = '#FFFFFF',
                       use_noise: bool = True) -> np.ndarray:
    """

        Args:
            img: orig img
            alpha_mask: mask of wall
            color: 16-bit hex string
            use_noise: Use noise in color generation
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
    result = img.copy() - img * alpha_mask + (color_image * alpha_mask)

    return result


def _change_polygon_texture(img: np.ndarray, alpha_mask: np.ndarray, pt1, pt2, texture:  np.ndarray,
                            texture_angle: float = 0, border_polygon: list=None) -> np.ndarray:
    """

        Args:
            pt1: first vanishing point
            pt2: second vanishing point
            img: orig img
            alpha_mask: mask of wall
            texture: new texture
            texture_angle: float
            border_polygon: if we know polygon, so dont need creating from maks
        Returns:
            Image with changed wall texture
        """
    texture = rotate_crop(texture, texture_angle)

    result = img.copy()
    if border_polygon is not None:
        polygon = border_polygon.copy()
        polygon.append(polygon[0])
    else:
        polygon = create_polygon(np.array(alpha_mask[..., 0], dtype=np.uint8))
    border = find_perspective_border(polygon, pt1, pt2, img.shape[:-1][::-1])
    matrix = cv2.getPerspectiveTransform(
        np.float32([[0, 0], [texture.shape[1] - 1, 0], [texture.shape[1] - 1, texture.shape[0] - 1],
                    [0, texture.shape[0] - 1]]), np.float32(border))
    warped_texture = cv2.warpPerspective(texture, matrix, (img.shape[1], img.shape[0]), cv2.INTER_LINEAR)
    result = result - result * alpha_mask + (warped_texture * alpha_mask)

    return result


def change_wall_polygons_material(img: np.ndarray, mask: np.ndarray, vps: list, polygons: list,
                                   apply_shadows: bool = True, object_mask: np.ndarray = None):
    """

        Args:
            img: orig img
            vps: list of 3 vanishing points
            mask: segmentation mask of image
            polygons: list of wall polygons [{points: [{'x': 0.00306, 'y': 0.0}, ...]
                                              material: str or np.ndarray
                                              (optional) layout_type: int}]
            apply_shadows: bool
        Returns:
            Image with changed wall texture
        """
    if object_mask is not None:
        replace_mask = object_mask
    else:
        replace_mask = mask
    result = img.copy()
    for polygon in polygons:
        wall_type = polygon.get('layout_type')
        if wall_type is None:
            wall_type = get_polygon_wall_type(polygon['points'], vps, points_scale=mask.shape)
        elif wall_type not in LAYOUT_WALL_INDEXES:
            continue
        formatted_polygon = [(int(point['x']*mask.shape[1]), int(point['y']*mask.shape[0]))
                            for point in polygon['points']]
        wall_mask = np.full(mask.shape, 255, dtype=np.uint8)
        wall_mask = cv2.drawContours(wall_mask, [np.array(formatted_polygon)], -1, (wall_type), -1)
        wall_mask = np.logical_and(wall_mask == wall_type, replace_mask == WALL_IDX).astype(np.uint8)
        if wall_mask.sum() == 0:
            continue
        alpha_mask = np.zeros([*wall_mask.shape, 3], dtype=np.uint8)
        alpha_mask[..., 0] = wall_mask
        alpha_mask[..., 1] = wall_mask
        alpha_mask[..., 2] = wall_mask
        if isinstance(polygon['material'], str):
            result = _change_polygon_color(img=result, alpha_mask=alpha_mask, color=polygon['material'])
        elif isinstance(polygon['material'], np.ndarray):
            pt1, pt2 = determine_vps(vps=vps, layout_type=GET_LAYOUT_TYPE[wall_type], points=formatted_polygon)
            result = _change_polygon_texture(img=result, alpha_mask=alpha_mask, pt1=pt1, pt2=pt2,
                                             texture=polygon['material'], border_polygon=formatted_polygon)
        else:
            raise TypeError(f"Wrong poligon material type: {type(polygon['material'])}")
        if apply_shadows:
            shadow_mask = (mask == WALL_IDX) * alpha_mask[..., 2]

            result = transfer_shadows(source_img=img, target_img=result, mask=shadow_mask, mask_target=1,
                                      dark_trash_scale=1.3, bright_trash_scale=1.5,
                                      blur_kernel=int(10 * max(img.shape) / 800))
    return result


def change_polygons_material(img: np.ndarray, vps: list, polygons: list, objects_polygons: list = None):
    """

            Args:
                img: orig img
                vps: list of 3 vanishing points
                polygons: list of wall polygons [{points: [{'x': 0.00306, 'y': 0.0}, ...]
                                                  material: str or np.ndarray
                                                  layout_type: int}]
                                                   * Layout types: {0: 'frontal', 1: 'left', 2: 'right'} - walls
                                                                   {3: 'floor', 4: 'celling'}
                                                                   {10: 'wall'} - indefinite wall
                objects_polygons: list of objects polygons to remove
                                                [{points: [{'x': 0.00306, 'y': 0.0}, ...]]

            Returns:
                Image with changed texture
            """
    result = img.copy()
    remove_mask = None
    if objects_polygons is not None:
        remove_mask = polygons_to_mask(objects_polygons, result.shape[:2], fill_value=1, default_polygon_value=0,
                                       polygon_key=None)
    for polygon in polygons:
        try:
            layout_type = polygon['layout_type']
        except KeyError:
            raise KeyError('It is required to specify layout_type in the polygon')
        try:
            layout_type = GET_LAYOUT_TYPE[layout_type]
        except KeyError:
            raise KeyError(f'Wrong layout type: {layout_type}')
        if layout_type == 'wall':
            layout_type = GET_LAYOUT_TYPE[get_polygon_wall_type(polygon['points'], vps, points_scale=img.shape[:2][::-1])]
        formatted_polygon = [(int(point['x'] * img.shape[1]), int(point['y'] * img.shape[0]))
                                                   for point in polygon['points']]
        polygon_mask = np.full(img.shape[:2], 0, dtype=np.uint8)
        polygon_mask = cv2.drawContours(polygon_mask,
                                        [np.array(formatted_polygon)], -1, (1), -1)
        if objects_polygons is not None:
            polygon_mask *= remove_mask
        alpha_mask = np.zeros([*polygon_mask.shape, 3], dtype=np.uint8)
        alpha_mask[..., 0] = polygon_mask
        alpha_mask[..., 1] = polygon_mask
        alpha_mask[..., 2] = polygon_mask
        if isinstance(polygon['material'], str):
            result = _change_polygon_color(img=result, alpha_mask=alpha_mask, color=polygon['material'])
        elif isinstance(polygon['material'], np.ndarray):
            pt1, pt2 = determine_vps(vps=vps, layout_type=layout_type,
                                     points=formatted_polygon)
            result = _change_polygon_texture(img=result, alpha_mask=alpha_mask, pt1=pt1, pt2=pt2,
                                             texture=polygon['material'], border_polygon=formatted_polygon)
        else:
            raise KeyError(f'Wrong material type: {type(polygon["material"])}')
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
