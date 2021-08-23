import os
from argparse import Namespace

import cv2
import numpy as np
import mmcv

from UprightNet.data.image_folder import inference_transform
from UprightNet.models.create_model import create_model
from lsun_room_master.trainer import core
from mmsegmentation.mmseg.apis import init_segmentor, inference_segmentor
import torchvision.transforms.functional as F
from PIL import Image

SEG_MODEL_CONFIG_PATH = os.path.join('mmsegmentation', 'configs', 'swin',
                                     'upernet_swin_tiny_patch4_window7_512x512_160k_ade20k.py')
SEG_MODEL_CHECKPOINT_PATH = os.path.join('mmsegmentation', 'checkpoints',
                                         'upernet_swin_tiny_patch4_window7_512x512.pth')

LAYOUT_MODEL_CHECKPOINT_PATH = os.path.join('lsun_room_master', 'ckpts',
                                            'model_retrained.ckpt')

PITCH_MODEL_CONFIG = Namespace(**{'mode': 'ResNet', 'dataset': 'interiornet', 'gpu_ids': None, 'isTrain': True,
                                  'checkpoints_dir': os.path.join(os.path.dirname(__file__), 'UprightNet', 'checkpoints'),
                                  'name': 'test_local'})

FLOOR_IDX = 3

seg_model = None
pitch_model = None
layout_model = None


def get_pitch_model(config:Namespace = PITCH_MODEL_CONFIG):
    global pitch_model
    if pitch_model is None:
        pitch_model = create_model(config, _isTrain=False)
    return pitch_model


def get_seg_model(config_path:str = SEG_MODEL_CONFIG_PATH, checkpoint_path:str = SEG_MODEL_CHECKPOINT_PATH):
    global seg_model
    if seg_model is None:
        seg_model = init_segmentor(config_path, checkpoint_path, device='cpu')
    return seg_model


def get_layout_model(checkpoint_path:str = LAYOUT_MODEL_CHECKPOINT_PATH):
    global layout_model
    if layout_model is None:
        layout_model = core.LayoutSeg.load_from_checkpoint(checkpoint_path, backbone='resnet101')
    return layout_model


def predict(image_path: str) -> (np.ndarray, float):
    seg_model = get_seg_model()
    pitch_model = get_pitch_model()
    img = mmcv.imread(image_path)

    result = inference_segmentor(seg_model, img)[0]
    result_floor = np.where(result==FLOOR_IDX, result, 0)
    transformed_img = inference_transform({'img': img})
    _, _, pitch = pitch_model.infer_model(transformed_img.unsqueeze(0), 1)
    return (result_floor, pitch)


def predict_mask(image: (str, np.ndarray)) -> np.ndarray:
    seg_model = get_seg_model()
    if isinstance(image, str):
        img = mmcv.imread(image)
    else:
        img = image

    result = inference_segmentor(seg_model, img)[0]

    return result


def predict_layout(image: (str, np.ndarray)) -> np.ndarray:
    tensor = Image.fromarray(image).convert('RGB')
    shape = tensor.size
    tensor = F.to_tensor(tensor)
    tensor = F.resize(tensor, [320, 320], interpolation=F.InterpolationMode.BILINEAR)
    tensor = F.normalize(tensor, mean=0.5, std=0.5)
    _, outputs = get_layout_model()(tensor.unsqueeze(0).cpu())
    outputs = outputs.cpu()[0].numpy()
    outputs = cv2.resize(np.array(outputs, dtype=np.uint8), shape[:2])
    return outputs


if __name__ == "__main__":
    print(predict('UprightNet/demo/demo/ADE_val_00000118.jpg'))
