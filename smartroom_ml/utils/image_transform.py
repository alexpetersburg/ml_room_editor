import numpy as np
import cv2
import math


def warp_perspective(img: np.ndarray, alpha: float, dz: float = None, f: float = None, dy: float = 0, dx: float = 0) -> (np.ndarray, np.ndarray):
    """

    Args:
        f: Focal
        alpha: vertical rotate angle [0: 90]
        dz: distance from the virtual camera to the image. default max(img.shape) * 1.4
        dy: offset axis y
        dx: offset axis x

    Returns:
        (warp, trans)
        warp: img after transform
        trans: transformation matrix
    """
    alpha = (alpha - 90.) * np.pi / 180

    if dz is None:
        dz = max(img.shape) * 1.4
    if f is None:
        f = max(img.shape)
    w = img.shape[0]
    h = img.shape[1]

    A1 = np.array([[1, 0, -w / 2],
                   [0, 1, -h / 2],
                   [0, 0, 0],
                   [0, 0, 1]])

    RX = np.array([[1, 0, 0, 0],
                   [0, np.cos(alpha), -np.sin(alpha), 0],
                   [0, np.sin(alpha), np.cos(alpha), 0],
                   [0, 0, 0, 1]])
    RY = np.array([[np.cos(0), 0, -np.sin(0), 0],
                   [0, 1, 0, 0],
                   [np.sin(0), 0, np.cos(0), 0],
                   [0, 0, 0, 1]])
    RZ = np.array([[np.cos(0), -np.sin(0), 0, 0],
                   [np.sin(0), np.cos(0), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]]);
    R = RX.dot(RY).dot(RZ)

    T = np.array([[1, 0, 0, dx],
                  [0, 1, 0, dy],
                  [0, 0, 1, dz],
                  [0, 0, 0, 1]])
    A2 = np.array([[f, 0, w / 2, 0],
                   [0, f, h / 2, 0],
                   [0, 0, 1, 0]])
    trans = R.dot(A1)
    trans = T.dot(trans)
    trans = A2.dot(trans)
    warp = cv2.warpPerspective(img, trans, (img.shape[1], img.shape[0]), cv2.INTER_LANCZOS4)
    return (warp, trans)


def compute_angle(pitch):
    if pitch < -0.4:
        angle = max((70 - 70 * (1 + pitch)), 15)
    else:
        angle = max((80 - 70 * (1 + pitch)), 15)
    return angle


def autocrop(image, threshold=0):
    """Crops any edges below or equal to threshold

    Crops blank image to 1x1.

    Returns cropped image.

    """
    if len(image.shape) == 3:
        flatImage = np.max(image, 2)
    else:
        flatImage = image
    assert len(flatImage.shape) == 2

    rows = np.where(np.max(flatImage, 0) > threshold)[0]
    if rows.size:
        cols = np.where(np.max(flatImage, 1) > threshold)[0]
        image = image[cols[0]: cols[-1] + 1, rows[0]: rows[-1] + 1]
    else:
        image = image[:1, :1]

    return image


def crop_top_image(image, threshold=0):
    """Crops top edge below or equal to threshold

    Crops blank image to 1x1.

    Returns cropped image.

    """
    if len(image.shape) == 3:
        flatImage = np.max(image, 2)
    else:
        flatImage = image
    assert len(flatImage.shape) == 2

    rows = np.where(np.max(flatImage, 0) > threshold)[0]
    if rows.size:
        cols = np.where(np.max(flatImage, 1) > threshold)[0]
        image = image[cols[0]:, :]
    else:
        image = image[:1, :1]

    return image


def crop_left_right_borders(image, threshold=0):
    """Crops top edge below or equal to threshold

        Crops blank image to 1x1.

        Returns cropped image.

        """
    if len(image.shape) == 3:
        flatImage = np.max(image, 2)
    else:
        flatImage = image
    assert len(flatImage.shape) == 2
    rows = np.where(flatImage[0] > threshold)[0]
    if rows.size:
        image = image[:, rows[0]:rows[-1]]
    else:
        image = image[:1, :1]

    return image


def rotate_image(image, angle):
    """
    Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
    (in degrees). The returned image will be large enough to hold the entire
    new image, with a black background
    """

    # Get the image size
    # No that's not an error - NumPy stores image matricies backwards
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    # Apply the transform
    result = cv2.warpAffine(
        image,
        affine_mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR
    )

    return result


def largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell
    """

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )


def crop_around_center(image, width, height):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if width > image_size[0]:
        width = image_size[0]

    if height > image_size[1]:
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]


def rotate_crop(img: np.ndarray, angle: float) -> np.ndarray:
    image_rotated = rotate_image(img, angle)
    image_rotated_cropped = crop_around_center(
        image_rotated,
        *largest_rotated_rect(
            img.shape[1],
            img.shape[0],
            math.radians(angle)
        )
    )
    return image_rotated_cropped


def demo():
    """
    Demos the largest_rotated_rect function
    """

    image = cv2.imread("../demo/demo/ADE_val_00000118.jpg")
    image_height, image_width = image.shape[0:2]

    cv2.imshow("Original Image", image)

    key = cv2.waitKey(0)
    if key == ord("q") or key == 27:
        exit()

    for i in np.arange(0, 360, 0.5):
        image_orig = np.copy(image)
        print(image_orig.shape)
        image_rotated = rotate_image(image, i)
        print(image_rotated.shape)
        image_rotated_cropped = crop_around_center(
            image_rotated,
            *largest_rotated_rect(
                image_width,
                image_height,
                math.radians(i)
            )
        )

        key = cv2.waitKey(2)
        if(key == ord("q") or key == 27):
            exit()

        cv2.imshow("Original Image", image_orig)
        cv2.imshow("Rotated Image", image_rotated)
        cv2.imshow("Cropped Image", image_rotated_cropped)


def find_layout_polygons(layout_mask, blur_kernel=15, approx_strength=0.01):
    def pad_with(vector, pad_width, iaxis, kwargs):
        pad_value = kwargs.get('padder', 10)
        vector[:pad_width[0]] = pad_value
        vector[-pad_width[1]:] = pad_value

    layout = np.pad(layout_mask, 2, pad_with, padder=-1)
    layout_segments = {}
    for plane_class in np.unique(layout_mask):
        wall = ((layout == plane_class)).astype(np.uint8)

        # find largest connected area
        conn = cv2.connectedComponents(wall)
        u, count = np.unique(conn[1], return_counts=True)
        count_sort_ind = np.argsort(-count)
        wall = (conn[1] == np.delete(u[count_sort_ind], np.where(u[count_sort_ind] == 0))[0]).astype(np.uint8)

        wall = cv2.GaussianBlur(wall, (blur_kernel, blur_kernel), 0)

        contours, _ = cv2.findContours(wall, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour = contours[0]
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, approx_strength * perimeter, True)
        approx = [{'point': [max(0, min(point[0][0] - 2, layout_mask.shape[1])),
                             max(0, min(point[0][1] - 2, layout_mask.shape[0]))], 'point_index': i,
                   'point_classes': [int(plane_class)]}
                  for i, point in enumerate(approx)]
        layout_segments[int(plane_class)] = approx

    layout_segments = fuse_points(layout_segments, max(layout_mask.shape) * 0.01)
    out_polygons = []
    for layout_class, points in layout_segments.items():
        out_polygons.append({'points': [{'x': float(point['point'][0] / layout_mask.shape[1]),
                                          'y': float(point['point'][1] / layout_mask.shape[0]),
                                          'point_classes': point['point_classes']} for point in points]
    #     print(layout_segments[1])
    #     print(layout_segments[2])
    return out_polygons


def fuse_points(layout_segments, d):
    def dist2(p1, p2):
        return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
    points = [item for sublist in layout_segments.values() for item in sublist]
    d2 = d * d
    n = len(points)
    taken = [False] * n
    for i in range(n):
        if not taken[i]:
            count = 1
            point = points[i]
            update_indexes = [[point['point_index'], point['point_classes'][0]]]
            classes = point['point_classes']
            taken[i] = True
            for j in range(i + 1, n):
                if taken[j]:
                    continue
                if dist2(points[i]['point'], points[j]['point']) < d2:
                    update_indexes.append([points[j]['point_index'], points[j]['point_classes'][0]])
                    classes.append(int(points[j]['point_classes'][0]))
                    point['point'][0] += points[j]['point'][0]
                    point['point'][1] += points[j]['point'][1]
                    count += 1
                    taken[j] = True
            point['point'][0] /= count
            point['point'][1] /= count
            for i, layout_class in update_indexes:
                layout_segments[int(layout_class)][i]['point'] = point['point']
                layout_segments[int(layout_class)][i]['point_classes'] = list(set(classes))
    return layout_segments


if __name__ == "__main__":
    image = cv2.imread("../demo/demo/ADE_val_00000118.jpg")
    result = rotate_crop(image, 30)
    demo()