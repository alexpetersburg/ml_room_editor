#!/usr/bin/env python3
"""Compute vanishing points using corase-to-fine method on the evaluation dataset.
Usage:
    eval.py [options] <yaml-config> <checkpoint>
    eval.py ( -h | --help )

Arguments:
   <yaml-config>                 Path to the yaml hyper-parameter file
   <checkpoint>                  Path to the checkpoint

Options:
   -h --help                     Show this screen
   -d --devices <devices>        Comma seperated GPU devices [default: 0]
   -o --output <output>          Path to the output AA curve [default: error.npz]
   --dump <output-dir>           Optionally, save the vanishing points to npz format.
                                 The coordinate of VPs is in the camera space, see
                                 `to_label` and `to_pixel` in neurvps/models/vanishing_net.py
                                 for more details.
   --noimshow                    Do not show result
"""

import os
import sys
import math
import shlex
import pprint
import random
import os.path as osp
import threading
import subprocess

import cv2
import numpy as np
import torch
import matplotlib as mpl
import skimage.io
import numpy.linalg as LA
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from tqdm import tqdm
from docopt import docopt

import neurvps
import neurvps.models.vanishing_net as vn
from neurvps.config import C, M
from neurvps.utils import sample_sphere

# torch.cuda.is_available = lambda: False


def AA(x, y, threshold):
    index = np.searchsorted(x, threshold)
    x = np.concatenate([x[:index], [threshold]])
    y = np.concatenate([y[:index], [threshold]])
    return ((x[1:] - x[:-1]) * y[:-1]).sum() / threshold


def neurvps_predict(image, model):
    x_bias = y_bias = 0
    h, w, _ = image.shape
    if h > w:
        bias = int((h - w) / 2)
        image = image[bias:bias + w, :, :]
        y_bias = bias
    elif w > h:
        bias = int((w - h) / 2)
        image = image[:, bias:bias + h, :]
        x_bias = bias
    scale = 512 / image.shape[0]
    image = cv2.resize(image, (512, 512))

    image = np.rollaxis(image.astype(np.float), 2).copy()
    image = torch.tensor(image).float()
    image = torch.unsqueeze(image, 0)
    device = torch.device('cuda')
    image = image.to(device)
    input_dict = {"image": image, "test": True}
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
    pixel = [vn.to_pixel(v) for v in vpts_pd]
    pixel = [(point[0] / scale + x_bias, point[1] / scale + y_bias) for point in pixel]
    return pixel

def main():
    args = docopt(__doc__)
    config_file = args["<yaml-config>"]
    C.update(C.from_yaml(filename=config_file))
    C.model.im2col_step = 32  # override im2col_step for evaluation
    M.update(C.model)
    pprint.pprint(C, indent=4)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    device_name = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = args["--devices"]
    if torch.cuda.is_available():
        device_name = "cuda"
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(0)
        print("Let's use", torch.cuda.device_count(), "GPU(s)!")
    else:
        print("CUDA is not available")
    device = torch.device(device_name)

    if M.backbone == "stacked_hourglass":
        model = neurvps.models.hg(
            planes=64, depth=M.depth, num_stacks=M.num_stacks, num_blocks=M.num_blocks
        )
    else:
        raise NotImplementedError

    checkpoint = torch.load(args["<checkpoint>"], map_location=device)
    model = neurvps.models.VanishingNet(
        model, C.model.output_stride, C.model.upsample_scale
    )
    model = model.to(device)
    model = torch.nn.DataParallel(
        model, device_ids=list(range(args["--devices"].count(",") + 1))
    )  # ???
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if args["--dump"] is not None:
        os.makedirs(args["--dump"], exist_ok=True)

    err = []
    n = C.io.num_vpts
    for path in os.listdir('demo'):
        if '12_1' not in path:
            continue
        image = skimage.io.imread(os.path.join('demo', path))[:, :, :3]
        image = cv2.resize(image, None, fx=2, fy=2)
        x_bias = y_bias = 0
        h, w, _ = image.shape
        if h > w:
            bias = int((h - w) / 2)
            image = image[bias:bias + w, :, :]
            y_bias = bias
        elif w > h:
            bias = int((w - h)/2)
            image = image[:, bias:bias+h, :]
            x_bias = bias
        image = cv2.resize(image, (512, 512))
        orig_img = image.copy()

        # cv2.imshow('asd', image)
        # cv2.waitKey()
        image = np.rollaxis(image.astype(np.float), 2).copy()
        image = torch.tensor(image).float()
        image = torch.unsqueeze(image, 0)
        image = image.to(device)
        input_dict = {"image": image, "test": True}
        vpts = sample_sphere(np.array([0, 0, 1]), np.pi / 2, 64)
        input_dict["vpts"] = vpts
        with torch.no_grad():
            score = model(input_dict)[:, -1].cpu().numpy()
        index = np.argsort(-score)
        candidate = [index[0]]
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
        print('pred: ', vpts_pd)
        pixel = [vn.to_pixel(v) for v in vpts_pd]
        pixel = [(point[0]+x_bias, point[1]+y_bias) for point in pixel]
        print('pixel:', pixel)
        plt.axis([512, 0, 512, 0])
        plt.imshow(orig_img)
        cc = ["blue", "cyan", "orange"]
        print(C.io.focal_length)
        # C.io.focal_length = 2
        for c, w in zip(cc, vpts_pd):
            x = w[0] / w[2] * C.io.focal_length * 256 + 256
            y = -w[1] / w[2] * C.io.focal_length * 256 + 256
            print(x, y)
            plt.scatter(x, y, color=c)
            for xy in np.linspace(0, 512, 10):
                plt.plot(
                    [x, xy, x, xy, x, 0, x, 511],
                    [y, 0, y, 511, y, xy, y, xy],
                    color=c,
                )
        plt.savefig(os.path.join('result', path),dpi=300)
        plt.clf()



def sample_sphere(v, alpha, num_pts):
    v1 = orth(v)
    v2 = np.cross(v, v1)
    v, v1, v2 = v[:, None], v1[:, None], v2[:, None]
    indices = np.linspace(1, num_pts, num_pts)
    phi = np.arccos(1 + (math.cos(alpha) - 1) * indices / num_pts)
    theta = np.pi * (1 + 5 ** 0.5) * indices
    r = np.sin(phi)
    return (v * np.cos(phi) + r * (v1 * np.cos(theta) + v2 * np.sin(theta))).T


def orth(v):
    x, y, z = v
    o = np.array([0.0, -z, y] if abs(x) < abs(y) else [-z, 0.0, x])
    o /= LA.norm(o)
    return o


if __name__ == "__main__":
    main()
