import os
from argparse import Namespace

import cv2
import numpy as np
import mmcv
import torch

from smartroom_ml.UprightNet.data.image_folder import inference_transform
from smartroom_ml.UprightNet.models.create_model import create_model
from smartroom_ml.lsun_room_master.trainer import core
from smartroom_ml.mmsegmentation.mmseg.apis import init_segmentor, inference_segmentor
import torchvision.transforms.functional as F
from PIL import Image


from smartroom_ml import neurvps
import smartroom_ml.neurvps.models.vanishing_net as vn
from smartroom_ml.neurvps.config import C, M
from smartroom_ml.neurvps.utils import sample_sphere, sort_vps
import matplotlib.pyplot as plt

SMARTROOM_DIR = os.path.dirname(os.path.abspath(__file__))
SEG_MODEL_CONFIG_PATH = os.path.join(SMARTROOM_DIR,'mmsegmentation', 'configs', 'swin',
                                     'upernet_swin_base_patch4_window7_512x512_160k_ade20k.py')
SEG_MODEL_CHECKPOINT_PATH = os.path.join(SMARTROOM_DIR,'mmsegmentation', 'checkpoints',
                                         'upernet_swin_base_patch4_window7_512x512.pth')

VPS_MODEL_CONFIG_PATH = os.path.join(SMARTROOM_DIR, 'neurvps_utils', 'logs', 'ScanNet', 'config.yaml')
VPS_MODEL_CHECKPOINT_PATH = os.path.join(SMARTROOM_DIR,'neurvps_utils', 'logs', 'ScanNet', 'better-result.pth.tar')


LAYOUT_MODEL_CHECKPOINT_PATH = os.path.join(SMARTROOM_DIR,'lsun_room_master', 'ckpts',
                                            'model_retrained.ckpt')

PITCH_MODEL_CONFIG = Namespace(**{'mode': 'ResNet', 'dataset': 'interiornet', 'gpu_ids': None, 'isTrain': True,
                                  'checkpoints_dir': os.path.join(SMARTROOM_DIR, 'UprightNet', 'checkpoints'),
                                  'name': 'test_local'})

FLOOR_IDX = 3

seg_model = None
pitch_model = None
layout_model = None
vps_model = None
device = None


def get_device():
    global device
    if device is None:
        device_name = "cpu"
        if torch.cuda.is_available():
            device_name = "cuda"
            torch.backends.cudnn.deterministic = True
            torch.cuda.manual_seed(0)
        device = torch.device(device_name)
    return device


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


def get_vps_model(config_path:str = VPS_MODEL_CONFIG_PATH, checkpoint_path:str = VPS_MODEL_CHECKPOINT_PATH):
    global vps_model
    if vps_model is None:
        C.update(C.from_yaml(filename=config_path))
        C.model.im2col_step = 32  # override im2col_step for evaluation
        M.update(C.model)

        device = get_device()

        if M.backbone == "stacked_hourglass":
            model = neurvps.models.hg(
                planes=64, depth=M.depth, num_stacks=M.num_stacks, num_blocks=M.num_blocks
            )
        else:
            raise NotImplementedError

        checkpoint = torch.load(checkpoint_path)
        model = neurvps.models.VanishingNet(
            model, C.model.output_stride, C.model.upsample_scale
        )
        model = model.to(device)
        model = torch.nn.DataParallel(
            model, device_ids=[0]
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        vps_model = model
    return vps_model


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


def predict_vps(image: (str, np.ndarray)) -> np.ndarray:
    model = get_vps_model()
    if isinstance(image, str):
        image = mmcv.imread(image)[:, :, :3]
    x_bias = y_bias = 0
    h, w, _ = image.shape
    tensor_image = image.copy()
    if h > w:
        bias = int((h - w) / 2)
        tensor_image = tensor_image[bias:bias + w, :, :]
        y_bias = bias
    elif w > h:
        bias = int((w - h) / 2)
        tensor_image = tensor_image[:, bias:bias + h, :]
        x_bias = bias
    scale = 512 / tensor_image.shape[0]
    tensor_image = cv2.resize(tensor_image, (512, 512))
    tensor_image = np.rollaxis(tensor_image.astype(np.float), 2).copy()
    tensor_image = torch.tensor(tensor_image).float()
    tensor_image = torch.unsqueeze(tensor_image, 0)
    tensor_image = tensor_image.to(get_device())
    input_dict = {"image": tensor_image, "test": True}
    vpts = sample_sphere(np.array([0, 0, 1]), np.pi / 2, 64)
    input_dict["vpts"] = vpts
    with torch.no_grad():
        score = model(input_dict)[:, -1].cpu().numpy()
    index = np.argsort(-score)
    candidate = [index[0]]
    n = C.io.num_vpts
    for i in index[1:]:
        if len(candidate) == n:
            break
        dst = np.min(np.arccos(np.abs(vpts[candidate] @ vpts[i])))
        if dst < np.pi / n:
            continue
        candidate.append(i)
    vpts_pd = vpts[candidate]

    for res in range(1, len(M.multires)):
        vpts = [sample_sphere(vpts_pd[vp], M.multires[-res], 64) for vp in range(n)]
        input_dict["vpts"] = np.vstack(vpts)
        with torch.no_grad():
            score = model(input_dict)[:, -res - 1].cpu().numpy().reshape(n, -1)
        for i, s in enumerate(score):
            vpts_pd[i] = vpts[i][np.argmax(s)]
    pixel = [vn.to_pixel(v)[::-1] for v in vpts_pd]
    pixel = [(point[0] / scale + x_bias, point[1] / scale + y_bias) for point in pixel]
    pixel = sort_vps(pixel)
    return pixel


if __name__ == "__main__":
    print(predict('UprightNet/demo/demo/ADE_val_00000118.jpg'))
