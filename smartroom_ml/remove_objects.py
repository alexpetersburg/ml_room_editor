import cv2
import numpy as np


def find_objects(mask: np.ndarray, target_classes: list) -> np.ndarray:
    result_mask = np.zeros_like(mask)
    active_object = 1
    for i in range(result_mask.shape[0]):
        for j in range(result_mask.shape[1]):
            if mask[i, j] not in target_classes:
                if j > 0 and result_mask[i, j - 1] != 0:
                    active_object = result_mask.max() + 1
                result_mask[i, j] = 0
            elif mask[i, j] in target_classes:
                if i > 0 and result_mask[i - 1, j] != 0:
                    if result_mask[i - 1, j] < active_object:
                        result_mask = np.where(result_mask == active_object, result_mask[i - 1, j], result_mask)
                        active_object = result_mask[i - 1, j]
                    elif result_mask[i -1, j] > active_object:
                        result_mask = np.where(result_mask == result_mask[i - 1, j], active_object, result_mask)
                result_mask[i, j] = active_object
        active_object = result_mask.max() + 1
    return result_mask


def remove_object_from_mask(mask, object_mask, layout, floor_idx, wall_idx):
    mapper = {0: wall_idx, 1: wall_idx, 2:wall_idx, 3:floor_idx, 4: -1}
    floor_wall_layout = np.copy(layout)
    for k, v in mapper.items():
        floor_wall_layout[layout == k] = v
    replacement_mask = object_mask * (floor_wall_layout != -1)
    return mask - mask * replacement_mask + replacement_mask * floor_wall_layout