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
5) [Download](https://github.com/SwinTransformer/storage/releases/download/v1.0.1/upernet_swin_tiny_patch4_window7_512x512.pth) segmentation model and paste to `mmsegmentation/checkpoints` folder
6) [Download](https://drive.google.com/file/d/1dlzzHxZeakkYSFfrWrO519Wc9W0GC2L_/view?usp=sharing) pitch model and unzip to `UprightNet` folder
# Usage
```python
from inference import predict
segmentation_mask, pitch = predict(image_path)
```