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
cd XiaohuLuVPDetection
python setup.py build
python setup.py install
```
   
6) [Download](https://github.com/SwinTransformer/storage/releases/download/v1.0.1/upernet_swin_tiny_patch4_window7_512x512.pth) segmentation model and paste to `mmsegmentation/checkpoints` folder
7) [Download](https://drive.google.com/file/d/1dlzzHxZeakkYSFfrWrO519Wc9W0GC2L_/view?usp=sharing) pitch model and unzip to `UprightNet` folder
8) [Download](https://drive.google.com/file/d/1fgAZbE70v8ghTZaj4WSHzSlNb5NJreus/view?usp=sharing) room layout model and paste to `lsun_room_master/ckpts` folder
# Usage
```python
from inference import predict, predict_mask
segmentation_mask, pitch = predict(image_path)
segmentation_mask = predict_mask(image_path) # predict_mask(image)

-----------------------------------------------

from texture_transform_vps import change_floor_texture, change_wall_color, change_wall_texture
result_floor = change_floor_texture(img=img, mask=mask, texture=texture, texture_angle=0)

# change wall color
result_wall = change_wall_color(img=img, mask=mask, color='#A91D11')

# change wall texture
result_wall = change_wall_texture(img=img, mask=mask, texture=wall_texture)
```