from subprocess import check_output
import json
from typing import Optional
import os

FILE_DIR = os.path.dirname(os.path.abspath(__file__))


def compute_camera_params(height: float, width: float, vp1: tuple, vp2: tuple) -> Optional[dict]:
    try:
        result = check_output(['node', f"{os.path.join(FILE_DIR, 'fspy', 'self_solver.js')}", str(height), str(width),
                               str(vp1[0]), str(vp1[1]), str(vp2[0]), str(vp2[1])])
        return json.loads(result)
    except Exception:
        return None


if __name__ == '__main__':
    h, w, x1, y1, x2, y2 = 1170, 780, -118.47391956827687, 384.3497574239535, 1080.5994814938165, 397.6670855398042
    print(compute_camera_params(h,w, (x1, y1), (x2, y2)))