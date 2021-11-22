from subprocess import check_output
import json
from typing import Optional
import os

FILE_DIR = os.path.dirname(os.path.abspath(__file__))


def compute_camera_params(height: float, width: float, vp1: tuple, vp2: tuple) -> Optional[dict]:
    result = None
    try:
        result = check_output(['node', f"{os.path.join(FILE_DIR, 'fspy', 'self_solver.js')}", str(width), str(height),
                               str(vp1[0]), str(vp1[1]), str(vp2[0]), str(vp2[1])])
        return json.loads(result)
    except Exception:
        if result:
            print(result)
        return None


if __name__ == '__main__':
    h, w, x1, y1, x2, y2 = 1200, 1800, 8632.225520946467, 639.3851153323312, 583.0174447906775,  473.5652372431185
    print(compute_camera_params(h,w, (x1, y1), (x2, y2)))