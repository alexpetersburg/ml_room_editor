from __future__ import division
import time
import torch
import numpy as np
from torch.autograd import Variable
import models.networks
from options.inference_options import InferenceOptions
import sys
from data.data_loader import *
from models.create_model import create_model
import random
from tensorboardX import SummaryWriter
import os, json, cv2, random

EVAL_BATCH_SIZE = 8
opt = InferenceOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch

eval_num_threads = 3
test_data_loader = CreateInferenceDataLoader(opt, opt.img_path,
                                                False, EVAL_BATCH_SIZE,
                                                eval_num_threads)
test_dataset = test_data_loader.load_data()
test_data_size = len(test_data_loader)

model = create_model(opt, _isTrain=False)
model.switch_to_train()

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
global_step = 0

def infer_model_on_image(model, dataset, global_step):
    rot_e_list = []
    roll_e_list = []
    pitch_e_list = []

    count = 0.0

    model.switch_to_eval()

    count = 0

    for i, data in enumerate(dataset):
        stacked_img = data[0]
        targets = data[1]

        est_up_n, pred_roll, pred_pitch = model.infer_model(stacked_img, targets)

        print('**********************************')
        print('**********************************')
        print('**********************************')
        print('CAM UP VEC3', est_up_n)
        print('PRED ROLL', pred_roll)
        print('PRED PITCH', pred_pitch)
        print('**********************************')
        print('**********************************')
        print('**********************************')

infer_model_on_image(model, test_dataset, global_step)

