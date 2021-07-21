import cv2
import os
import numpy as np

from UprightNet.utils.image_transform import compute_angle, rotate_crop, warp_perspective, autocrop, \
    crop_left_right_borders, crop_top_image
from inference import predict

FLOOR_IDX = 3


def change_floor_texture(img: np.ndarray, mask: np.ndarray,
                         pitch: float, texture: np.ndarray, texture_angle=0) -> np.ndarray:
    """

    Args:
        img: orig img
        mask: seg mask of floor
        pitch: camera angle
        texture: new floor texture
        texture_angle: angle of texture rotation

    Returns:
        Image with changed floor texture
    """
    # Transform texture
    angle = compute_angle(pitch)
    texture = rotate_crop(texture, texture_angle)
    warped_texture, _ = warp_perspective(texture, angle)
    warped_texture = autocrop(warped_texture)
    warped_texture = crop_left_right_borders(warped_texture)

    # crop top part of mask if no floor
    cropped_mask = crop_top_image(mask)

    # resize texture to cropped_mask shape
    texture_h = warped_texture.shape[0] * cropped_mask.shape[1] / warped_texture.shape[1]
    texture_w = cropped_mask.shape[1]
    if texture_h > cropped_mask.shape[0]:
        warped_texture = cv2.resize(warped_texture, (int(texture_w),
                                                     int(texture_h)))[-cropped_mask.shape[0]:, :]
    else:
        texture_h = cropped_mask.shape[0]
        texture_w = warped_texture.shape[1] * cropped_mask.shape[0] / warped_texture.shape[0]
        warped_texture = cv2.resize(warped_texture, (int(texture_w),
                                                     int(texture_h)))[:, :cropped_mask.shape[1]]

    # convert segmentation mask to alpha mask
    alpha_mask = np.zeros([*mask.shape, 3], dtype=np.uint8)
    alpha_mask[..., 0] = mask == FLOOR_IDX
    alpha_mask[..., 1] = mask == FLOOR_IDX
    alpha_mask[..., 2] = mask == FLOOR_IDX

    # apply alpha mask and add texture
    result = img.copy() - img * alpha_mask
    _ = np.zeros([*mask.shape, 3], dtype=np.uint8)
    _[-warped_texture.shape[0]:, result.shape[1] - warped_texture.shape[1]:] = warped_texture
    warped_texture = _
    result[-warped_texture.shape[0]:, result.shape[1] - warped_texture.shape[1]:] += \
        warped_texture * alpha_mask

    return result


if __name__ == "__main__":
    demo_img_path = os.path.join('UprightNet', 'demo', 'demo', 'ADE_val_00000118.jpg')
    img = cv2.imread(demo_img_path)
    texture = cv2.imread(os.path.join('UprightNet', 'demo', 'tile.jpg'))

    mask, pitch = predict(demo_img_path)
    result = change_floor_texture(img=img, mask=mask, pitch=pitch, texture=texture, texture_angle=-20)
    cv2.imshow('orig', img)
    cv2.imshow('result', result)
    cv2.waitKey(0)
