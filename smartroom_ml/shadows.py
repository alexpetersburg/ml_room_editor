import cv2
import numpy as np


def transfer_shadows(source_img: np.ndarray, target_img: np.ndarray, mask: np.ndarray,
                     mask_target: int = 3, dark_trash_scale: float = 1.5, bright_trash_scale: float = 1.5,
                     blur_kernel: int = 25, max_shadow_darkness: float = 0.3) -> np.ndarray:
    """

    Args:
        source_img:
        target_img:
        mask:
        mask_target: index of required class
        dark_trash_scale: how much more dark should be pixel to get into dark mask
        bright_trash_scale: how much more bright should be pixel to get into bright mask
        blur_kernel: blur kernel size
        max_shadow_darkness: maximum shading scale of target image

    Returns:

    """
    alpha_mask = np.zeros([*mask.shape, 3], dtype=np.uint8)
    alpha_mask[..., 0] = mask == mask_target
    alpha_mask[..., 1] = mask == mask_target
    alpha_mask[..., 2] = mask == mask_target

    # source_img = source_img * alpha_mask

    source_img = source_img
    gray_source_img = cv2.cvtColor(source_img, cv2.COLOR_RGB2GRAY)
    blur_gray_source_img = cv2.blur(gray_source_img, (blur_kernel, blur_kernel), 0)

    nonzero_blur_gray_source_img = blur_gray_source_img * alpha_mask[...,0]
    nonzero_blur_gray_source_img = nonzero_blur_gray_source_img.flatten()
    nonzero_blur_gray_source_img = nonzero_blur_gray_source_img[np.nonzero(nonzero_blur_gray_source_img)]
    hist, bin_edges = np.histogram(nonzero_blur_gray_source_img, bins=10)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.
    thresh = bin_centers[np.argmax(hist)]

    dark_mask = np.where(blur_gray_source_img < thresh / dark_trash_scale, blur_gray_source_img, 0) * alpha_mask[...,0]
    dark_mask = np.where(dark_mask == 0, 255, dark_mask)
    dark_mask = gray_source_img - gray_source_img * alpha_mask[...,0] + dark_mask * alpha_mask[...,0]
    dark_mask = cv2.blur(dark_mask, (int(blur_kernel), int(blur_kernel)), 0)
    dark_mask = dark_mask*alpha_mask[...,0]
    X_std = (dark_mask - dark_mask.min()) / (thresh - dark_mask.min())
    min_scale, max_scale = max_shadow_darkness, 1
    dark_mask = (X_std * (max_scale - min_scale) + min_scale)
    dark_mask = np.clip(dark_mask, 0, 1)

    bright_mask = np.where(blur_gray_source_img > thresh * bright_trash_scale, blur_gray_source_img, 0) * alpha_mask[...,0]
    bright_mask = cv2.blur(bright_mask, (int(blur_kernel), int(blur_kernel)), 0)
    bright_target_img = cv2.addWeighted(target_img, 1, cv2.cvtColor(bright_mask, cv2.COLOR_GRAY2RGB), 0.3, 0)

    hsv_bright_target_img = cv2.cvtColor(bright_target_img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_bright_target_img)
    v = np.array(v - v * alpha_mask[:, :, 0] + v * alpha_mask[:, :, 0] * dark_mask, dtype=np.uint8)
    final_hsv = cv2.merge((h, s, v))
    shadow_bright_target_img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    return shadow_bright_target_img


if __name__ == '__main__':
    import os
    from texture_transform_vps import change_floor_texture
    from inference import predict_mask


    for img_name in os.listdir(os.path.join('UprightNet', 'demo', 'input_imgs')):
        demo_img_path = os.path.join('UprightNet', 'demo', 'input_imgs', img_name)
        img = cv2.imread(demo_img_path)


        texture = cv2.imread(os.path.join('UprightNet', 'demo', 'wood.jpg'))

        mask = predict_mask(demo_img_path)
        result = change_floor_texture(img=img, mask=mask, texture=texture, texture_angle=0)

        shadows_result = transfer_shadows(source_img=img, target_img=result, mask=mask, mask_target=3)

        cv2.imwrite(os.path.join('perspective_via_vanishing_points', 'demo', 'result_with_shadows', img_name),
                    np.hstack([img, result, shadows_result]))
        # cv2.imwrite(os.path.join('perspective_via_vanishing_points', 'demo', 'result', 'mask.png'),
        #             mask)
        # cv2.imshow('orig', img)
        # cv2.imshow('result', result)
        # cv2.waitKey(0)
