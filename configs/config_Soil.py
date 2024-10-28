import os
import os.path as osp
import sys
import time
import numpy as np
from easydict import EasyDict as edict
import argparse

C = edict()
config = C
cfg = C

C.seed = 3407

remoteip = os.popen('pwd').read()
C.root_dir = os.path.abspath(os.path.join(os.getcwd(), './'))
C.abs_dir = osp.realpath(".")

# Dataset config
"""Dataset Path"""
C.dataset_name = 'Soil'
C.dataset_path = osp.join(C.root_dir, 'datasets', 'Soil')
C.rgb_root_folder = osp.join(C.dataset_path, 'Cutting')
C.rgb_format = '.png'
C.gt_root_folder = osp.join(C.dataset_path, 'Entire_Label')
C.gt_format = '.png'
C.gt_transform = False
# True when label 0 is invalid, you can also modify the function _transform_gt in dataloader.RGBXDataset
# True for most dataset valid, Faslse for MFNet(?)
C.x_root_folder = osp.join(C.dataset_path, 'Entire')
C.x_format = '.png'
C.x_is_single_channel = True # True for raw depth, thermal and aolp/dolp(not aolp/dolp tri) input
C.train_source = osp.join(C.dataset_path, "train.txt")
C.eval_source = osp.join(C.dataset_path, "test.txt")
C.is_test = False

# 读取训练集和测试集的实际数量
def count_lines(file_path):
    with open(file_path, 'r') as f:
        return sum(1 for line in f) -1 

C.num_train_imgs = count_lines(C.train_source)  # 更新为实际训练集数量
C.num_eval_imgs = count_lines(C.eval_source)    # 更新为实际测试集数量
# C.num_train_imgs = 10  #临时测试
# C.num_eval_imgs = 10    # 临时测试
C.num_classes = 6
C.class_names =  ['0', '1', '2', '3', '4','5']

"""Image Config"""
C.background = 5
C.image_height = 384
C.image_width = 384
C.norm_mean = np.array([0.485, 0.456, 0.406])
C.norm_std = np.array([0.229, 0.224, 0.225])

""" Settings for network, this would be different for each kind of model"""
C.backbone = 'sigma_tiny' # sigma_tiny / sigma_small / sigma_base
C.pretrained_model = None # do not need to change
C.decoder = 'MambaDecoder' # 'MLPDecoder'
C.decoder_embed_dim = 512
C.optimizer = 'AdamW'

"""Train Config"""
C.lr = 6e-5
C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 0.01
C.batch_size = 4
C.nepochs = 150
C.niters_per_epoch = C.num_train_imgs // C.batch_size  + 1
C.num_workers = 16
C.train_scale_array = [0.5, 0.75, 1, 1.25, 1.5, 1.75]
C.warm_up_epoch = 10

C.fix_bias = True
C.bn_eps = 1e-3
C.bn_momentum = 0.1

"""Eval Config"""
# C.eval_iter = 1
C.eval_stride_rate = 2 / 3
C.eval_scale_array = [1] # [0.75, 1, 1.25] # 
C.eval_flip = False # True # 
C.eval_crop_size = [384, 384] # [height weight]

"""Store Config"""
C.checkpoint_start_epoch = 1
C.checkpoint_step = 1

"""Path Config"""
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
add_path(osp.join(C.root_dir))

C.log_dir = osp.abspath('log_final/log_mfnet/' + 'log_' + C.dataset_name + '_' + C.backbone + '_' + 'cromb_conmb_cvssdecoder')
C.tb_dir = osp.abspath(osp.join(C.log_dir, "tb"))
C.log_dir_link = C.log_dir
C.checkpoint_dir = osp.abspath(osp.join(C.log_dir, "checkpoint"))

exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
C.log_file = C.log_dir + '/log_' + exp_time + '.log'
C.link_log_file = C.log_file + '/log_last.log'
C.val_log_file = C.log_dir + '/val_' + exp_time + '.log'
C.link_val_log_file = C.log_dir + '/val_last.log'

if __name__ == '__main__':
    print(config.nepochs)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-tb', '--tensorboard', default=False, action='store_true')
    args = parser.parse_args()

    if args.tensorboard:
        open_tensorboard()
