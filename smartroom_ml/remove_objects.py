import cv2
import numpy as np
from smartroom_ml.inference import predict_lama


def find_objects(mask: np.ndarray, target_classes: list) -> np.ndarray:
    orig_shape = mask.shape
    scale = 1

    if max(mask.shape) > 1000:
        scale = 1000 / max(mask.shape)
        mask = cv2.resize(mask.copy().astype(np.uint8), None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    result_mask = np.zeros_like(mask)
    active_object = 1
    active_object_mapper = {0: 0, 1: 1}
    for i in range(result_mask.shape[0]):
        for j in range(result_mask.shape[1]):
            if mask[i, j] not in target_classes:
                if j > 0 and result_mask[i, j - 1] != 0:
                    active_object = result_mask.max() + 1
                    active_object_mapper[active_object] = active_object
                result_mask[i, j] = 0
            elif mask[i, j] in target_classes:
                if i > 0 and result_mask[i - 1, j] != 0:
                    if result_mask[i - 1, j] < active_object:
                        active_object_mapper[active_object] = result_mask[i - 1, j]
                        # result_mask = np.where(result_mask == active_object, result_mask[i - 1, j], result_mask)
                        active_object = result_mask[i - 1, j]
                    elif result_mask[i -1, j] > active_object:
                        # result_mask = np.where(result_mask == result_mask[i - 1, j], active_object, result_mask)
                        if active_object_mapper.get(result_mask[i -1, j]) is not None and \
                                active_object_mapper.get(result_mask[i -1, j]) > active_object:
                            active_object_mapper[result_mask[i -1, j]] = active_object

                result_mask[i, j] = active_object
        active_object = result_mask.max() + 1
        active_object_mapper[active_object] = active_object
    for key in sorted(active_object_mapper.keys()):
        if key == active_object_mapper[key]:
            continue
        new_val = active_object_mapper.get(active_object_mapper[key])
        while new_val is not None:
            active_object_mapper[key] = new_val
            if new_val == active_object_mapper.get(new_val):
                break
            new_val = active_object_mapper.get(new_val)
    result_mask = np.vectorize(active_object_mapper.get)(result_mask)
    if scale != 1:
        result_mask = cv2.resize(result_mask.astype(np.uint8), orig_shape[::-1],  interpolation=cv2.INTER_NEAREST)

    return result_mask


def remove_object_from_mask(mask, object_mask, layout, floor_idx, wall_idx):
    mapper = {0: wall_idx, 1: wall_idx, 2:wall_idx, 3:floor_idx, 4: -1}
    floor_wall_layout = np.copy(layout)
    for k, v in mapper.items():
        floor_wall_layout[layout == k] = v
    replacement_mask = object_mask * (floor_wall_layout != -1)
    return mask - mask * replacement_mask + replacement_mask * floor_wall_layout


def remove_objects_lama(img, mask, object_mask, layout, floor_idx, wall_idx):
    new_img = predict_lama(img, object_mask)
    new_mask = remove_object_from_mask(mask, object_mask, layout, floor_idx, wall_idx)
    return new_img, new_mask