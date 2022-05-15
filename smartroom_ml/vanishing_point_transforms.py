from scipy.spatial import ConvexHull
from skimage.morphology import convex_hull_image
import numpy as np
from imantics import Mask
import itertools
from shapely.geometry import Polygon, LineString
from math import pi
import cv2

MAX_PERSPECTIVE_BORDER_ERROR = 0.05


def create_polygon(mask):
    mask_hull = convex_hull_image(mask)
    polygons = Mask(mask_hull).polygons()
    points = polygons.points[0]
    hull = ConvexHull(points)
    points = points[np.unique(hull.simplices)]
    points = np.vstack([points, points[0]])
    return points


def find_horizontal_intersection_lines(shapely_poly: Polygon, img_shape):
    result = []
    stride = 2
    previous_intersect = None
    previous_condidate = None
    for y in range(0, img_shape[1]+1, stride):
        condidate = (0, y)
        pt = (img_shape[0], y)
        line = LineString([condidate, pt])
        intersect = shapely_poly.intersects(line)
        if previous_intersect is None and intersect:
            result.append((condidate, pt))
            previous_intersect = intersect
        elif previous_intersect is not None and previous_intersect != intersect:
            if not previous_intersect:
                result.append((previous_condidate, pt))
            else:
                result.append((condidate, pt))
        previous_intersect = intersect
        previous_condidate = condidate
    if len(result) < 2:
        result.append((previous_condidate, (img_shape[0], img_shape[1])))
    return result


def find_intersection_lines(shapely_poly: Polygon, pt, img_shape):
    result = []
    pt_location = [None, None]
    if pt[0] < 0:
        pt_location[0] = 0  # left
    elif pt[0] > img_shape[0]:
        pt_location[0] = 2  # right
    else:
        pt_location[0] = 1  # mid

    if pt[1] < 0:
        pt_location[1] = 0  # top
    elif pt[1] > img_shape[1]:
        pt_location[1] = 2  # bot
    else:
        pt_location[1] = 1  # mid

    location_to_segments = {
        (0, 0): [1, 2],
        (0, 1): [0, 1, 2],
        (0, 2): [0, 1],
        (1, 0): [1, 2, 3],
        (1, 1): [0, 1, 2, 3],
        (1, 2): [3, 0, 1],
        (2, 0): [2, 3],
        (2, 1): [2, 3, 0],
        (2, 2): [3, 0],
    }
    stride = 2
    get_segment = {
        0: zip(range(0, img_shape[0], stride), itertools.repeat(0)),
        1: zip(itertools.repeat(img_shape[0] - 1), range(0, img_shape[1], stride)),
        2: zip(range(img_shape[0] - 1, 0, -stride), itertools.repeat(img_shape[1] - 1)),
        3: zip(itertools.repeat(0), range(img_shape[1] - 1, 0, -stride)),
    }
    previous_intersect = None
    previous_condidate = None
    for segment in location_to_segments[tuple(pt_location)]:
        for x, y in get_segment[segment]:
            condidate = (x, y)
            line = LineString([condidate, pt])
            intersect = shapely_poly.intersects(line)
            if previous_intersect is None and intersect and pt_location != [1, 1]:
                result.append((condidate, pt))
                previous_intersect = intersect
            elif previous_intersect is not None and previous_intersect != intersect:
                if previous_intersect == False:
                    result.append((previous_condidate, pt))
                else:
                    result.append((condidate, pt))
            previous_intersect = intersect
            previous_condidate = condidate
    if len(result) == 0:
        return [((0,0), (0, img_shape[1])),
                ((img_shape[0], 0), (img_shape[0],img_shape[1]))]
    if len(result) < 2:
        result.append((previous_condidate, pt))
    return result


def line_intersection(line1, line2, allow_parallel_intersection=False):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        if not allow_parallel_intersection:
            raise Exception('lines do not intersect')
        sorted_line = sorted([list(elem) for elem in line1], key=lambda x: x[0])
        if sorted_line[0][1] < sorted_line[1][1]:
            sorted_line[1][1] -= 1
        else:
            sorted_line[1][1] += 1
            return line_intersection(sorted_line, line2)

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return int(x), int(y)


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def vector_from_line(line: np.ndarray):
    vector = line[1]
    vector -= line[0]
    return vector


def angle_between(line1, line2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

    """
    v1 = vector_from_line(line1)
    v2 = vector_from_line(line2)
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    if angle > pi/2:
        angle = abs(angle-pi)
    return angle


def create_simple_border(polygon: np.ndarray):
    min_border = cv2.minAreaRect(polygon)
    return cv2.boxPoints(min_border).astype(int)


def find_perspective_border(polygon, pt1, pt2, img_shape):
    result = []
    if pt1 is None:
        if len(polygon) == 4:
            return polygon
        return create_simple_border(polygon)
    shapely_polygon = Polygon(polygon)
    pt1_lines = find_intersection_lines(shapely_polygon, pt1, img_shape)[:2]
    if pt2 is not None:
        pt2_lines = find_intersection_lines(shapely_polygon, pt2, img_shape)[:2]
    else:
        pt2_lines = find_horizontal_intersection_lines(shapely_polygon, img_shape)[:2]
    for pt1_line in pt1_lines:
        for pt2_line in pt2_lines:
            point = line_intersection(pt1_line, pt2_line)
            result.append(point)

    border_sorted = []
    border_max = sorted(result, key=lambda tup: tup[1])
    border_sorted.extend(sorted(border_max[:2], key=lambda tup: tup[0], reverse=True))
    border_sorted.extend(sorted(border_max[2:], key=lambda tup: tup[0]))
    result = border_sorted

    result_polygon = Polygon(result)
    diff_result = shapely_polygon.difference(result_polygon)
    if diff_result.area / shapely_polygon.area > MAX_PERSPECTIVE_BORDER_ERROR:
        result = create_simple_border(np.array(polygon))
    return result