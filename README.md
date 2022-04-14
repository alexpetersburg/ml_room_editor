# Installation
Tested on python 3.8
1) `git clone https://gitlab.com/stvml/floor_segmentation.git`
2) sh install.sh
   
or


2) `pip install -r requirements.txt`
3) Install mmcv: `mmcv-full==1.3.0 -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.8.0/index.html`
4) Install mmsegmentation: 
```shell
cd smartroom_ml/mmsegmentation
python setup.py install
cd ../..
```
5) Install VPDetection:
```shell
cd smartroom_ml/XiaohuLuVPDetection
python setup.py build
python setup.py install
cd ../..
```
6) [Download](https://drive.google.com/drive/folders/1okLUvvGEzqg-yvpwkjFBNsUtSrRDPT93?usp=sharing) ScanNet folder and paste to `smartroom_ml/neurvps_utils/logs` folder
7) [Download](https://github.com/SwinTransformer/storage/releases/download/v1.0.1/upernet_swin_base_patch4_window7_512x512.pth) segmentation model and paste to `smartroom_ml/mmsegmentation/checkpoints` folder
8) [Download](https://drive.google.com/file/d/1fgAZbE70v8ghTZaj4WSHzSlNb5NJreus/view?usp=sharing) room layout model and paste to `smartroom_ml/lsun_room_master/ckpts` folder
9) Install OneGan:
```shell
cd OneGan
python setup.py install
cd ..
```
10) 
```shell
cd smartroom_ml/lama
curl -L $(yadisk-direct https://disk.yandex.ru/d/ouP6l8VJ0HpMZg) -o big-lama.zip
unzip big-lama.zip
cd ../..
```
11) install package
```shell
pip install -e .
```
# Usage
```python
from smartroom_ml.inference import predict_mask, predict_layout, predict_neurvps
segmentation_mask = predict_mask(image)
layout_mask, layout_polygons = predict_layout(image) # layout_polygons: [{points: [{'x': 0.00306, 'y': 0.0}, ...]
                                                     #                    layout_type: int}]
vps = predict_neurvps(image)

-----------------------------------------------

from smartroom_ml.texture_transform_vps import change_floor_texture, change_wall_polygons_material, \
                                               change_polygons_material, \
                                               WALL_IDX, LAYOUT_FLOOR_INDEX
                                               
result_floor = change_floor_texture(img=img, mask=segmentation_mask, vps=vps, texture=texture, texture_angle=0,
                                    apply_shadows=True, replace_rug=True, object_mask=None, 
                                    layout=filter(lambda x: x['layout_type'] == LAYOUT_FLOOR_INDEX, layout_polygons).__next__())
                                                                                            

# change wall material
result_wall = change_wall_polygons_material(img=img, mask=segmentation_mask, vps=vps, polygons=polygons, 
                                             apply_shadows = True, object_mask = None)
"""
polygons:[{points: [{'x': 0.00306, 'y': 0.0}, ...]
           material: str or np.ndarray
           (optional) layout_type: int}]
"""


# change polygon material
result = change_polygons_material(img=img, vps=vps, polygons=polygons)
"""
polygons:[{points: [{'x': 0.00306, 'y': 0.0}, ...]
           material: str or np.ndarray
           layout_type: int}]
* Layout types: {0: 'frontal', 1: 'left', 2: 'right'} - walls
                {3: 'floor', 4: 'celling'}
                {10: 'wall'} - indefinite wall
"""

# remove objects
from smartroom_ml.remove_objects import find_objects, remove_object_from_mask
objects = find_objects(segmentation_mask, FURNITURE_IDXS, merge_objects)
specified_object_mask = remove_object_from_mask(mask=segmentation_mask, object_mask=objects==OBJ_IDX, layout=layout_mask,
                                                floor_idx=FLOOR_IDX,
                                                wall_idx=WALL_IDX)
all_object_mask = remove_object_from_mask(mask=segmentation_mask, object_mask=objects!=0, layout=layout_mask,
                                          floor_idx=FLOOR_IDX,
                                          wall_idx=WALL_IDX)

result_floor = change_floor_texture(img=img, mask=segmentation_mask, vps=vps, texture=texture, texture_angle=0,
                                    apply_shadows=True, replace_rug=True, object_mask=specified_object_mask)


# remove objects lama
from smartroom_ml.remove_objects import remove_objects_lama
result_img, object_mask = remove_objects_lama(img=img, mask=segmentation_mask, object_mask=objects!=0, 
                                              layout=layout_mask, floor_idx=FLOOR_IDX, wall_idx=WALL_IDX)


# calibrate treejs camera
from smartroom_ml.inference import predict_camera_parameters

params = predict_camera_parameters(img_height=h, img_width=w, vps=vps) 
print(params)
'''
    {'verticalFieldOfView': ..,
    'pos_arr': ..,
    'principalPoint': {"x": 0, "y": 0},
    'imageWidth': ..,
    'imageHeight': ..,
'''



```