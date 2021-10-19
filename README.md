# Installation
Tested on python 3.8
1) `git clone https://gitlab.com/stvml/floor_segmentation.git`
2) `pip install -r requirements.txt`
3) Install mmcv: `mmcv-full==1.3.0 -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.8.0/index.html`
4) Install mmsegmentation: 
```shell
cd mmsegmentation
python setup.py install
```
5) Install VPDetection:
```shell
cd smartroom_ml/XiaohuLuVPDetection
python setup.py build
python setup.py install
cd ../..
```
6) [Download](https://drive.google.com/drive/folders/1srniSE2JD6ptAwc_QRnpl7uQnB5jLNIZ) ScanNet folder and paste to `smartroom_ml/neurvps_utils/logs` folder
7) [Download](https://github.com/SwinTransformer/storage/releases/download/v1.0.1/upernet_swin_base_patch4_window7_512x512.pth) segmentation model and paste to `smartroom_ml/mmsegmentation/checkpoints` folder
8) [Download](https://drive.google.com/file/d/1fgAZbE70v8ghTZaj4WSHzSlNb5NJreus/view?usp=sharing) room layout model and paste to `smartroom_ml/lsun_room_master/ckpts` folder
9) Install OneGan:
```shell
git clone hhttps://github.com/leVirve/OneGan.git
cd OneGan
python setup.py install
cd ..
```
10) install package
```shell
python setup.py install
```
# Usage
```python
from smartroom_ml.inference import predict, predict_mask
segmentation_mask, pitch = predict(image_path)
segmentation_mask = predict_mask(image_path) # predict_mask(image)

-----------------------------------------------

from smartroom_ml.texture_transform_vps import change_floor_texture, change_wall_color, change_wall_texture
result_floor = change_floor_texture(img=img, mask=mask, texture=texture, texture_angle=0,
                                    apply_shadows=True, replace_rug=True, object_mask=None)

# change wall color
result_wall = change_wall_color(img=img, mask=mask, color='#A91D11', apply_shadows=True, object_mask=None)

# change wall texture
result_wall = change_wall_texture(img=img, mask=mask, texture=wall_texture, apply_shadows=True, object_mask=None)


from smartroom_ml.remove_objects import find_objects, remove_object_from_mask


```