from scipy.spatial import ConvexHull
from skimage.morphology import convex_hull_image
import numpy as np
from imantics import Mask
import itertools
from shapely.geometry import Polygon, LineString


def create_polygon(mask):
    mask_hull = convex_hull_image(mask)
    polygons = Mask(mask_hull).polygons()
    points = polygons.points[0]
    hull = ConvexHull(points)
    points = points[np.unique(hull.simplices)]
    points = np.vstack([points, points[0]])
    return points


def find_intersection_lines(polygon, pt, img_shape):
    shapely_poly = Polygon(polygon)
    result = []
    pt_location = [None, None]
    if pt[0] < 0:
        pt_location[0] = 0  # left
    elif pt[0] > img_shape[1]:
        pt_location[0] = 2  # right
    else:
        pt_location[0] = 1  # mid

    if pt[1] < 0:
        pt_location[1] = 0  # top
    elif pt[1] > img_shape[0]:
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
            if previous_intersect is None and intersect:
                result.append((condidate, pt))
                previous_intersect = intersect
            elif previous_intersect is not None and previous_intersect != intersect:
                if previous_intersect == False:
                    result.append((previous_condidate, pt))
                else:
                    result.append((condidate, pt))
            previous_intersect = intersect
            previous_condidate = condidate
    if len(result) < 2:
        result.append((previous_condidate, pt))
    return result


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return int(x), int(y)


def find_perspective_border(polygon, pt1, pt2, img_shape):
    # tmp = np.zeros(img_shape)
    result = []
    pt1_lines = find_intersection_lines(polygon, pt1, img_shape)[:2]
    pt2_lines = find_intersection_lines(polygon, pt2, img_shape)[:2]
    for pt1_line in pt1_lines:
        # tmp = cv2.line(tmp, tuple(pt1_line[0]), tuple(pt1_line[1]), (255, 255, 255), 3)
        for pt2_line in pt2_lines:
            # tmp = cv2.line(tmp, tuple(pt2_line[0]), tuple(pt2_line[1]), (255, 255, 255), 3)
            point = line_intersection(pt1_line, pt2_line)
            result.append(point)
    # pts = np.array(polygon, np.int32)
    # pts = pts.reshape((-1, 1, 2))
    # cv2.polylines(tmp, [pts], True, (0, 255, 255))
    # cv2.imshow('asd', tmp)
    # cv2.waitKey(0)
    return result