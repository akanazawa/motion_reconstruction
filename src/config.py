"""
Sets default args

Note all data format is NHWC because slim resnet wants NHWC.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from absl import flags
import os.path as osp
from os import makedirs
from glob import glob
from datetime import datetime
import json

import numpy as np

curr_path = osp.dirname(osp.abspath(__file__))
model_dir = osp.join(curr_path, '..', 'models')
if not osp.exists(model_dir):
    print('Fix path to models/')
    import ipdb
    ipdb.set_trace()
SMPL_MODEL_PATH = osp.join(model_dir, 'neutral_smpl_with_cocoplus_reg.pkl')
SMPL_FACE_PATH = osp.join(curr_path, 'tf_smpl', 'smpl_faces.npy')

# Default pred-trained model path for the demo.
PRETRAINED_MODEL = '/home/kanazawa/projects/hmr_sfv/models/Feb12_2100_save75_model.ckpt-667589'
# PRETRAINED_MODEL = osp.join(model_dir, 'model.ckpt-667589')

flags.DEFINE_string('smpl_model_path', SMPL_MODEL_PATH,
                    'path to the neurtral smpl model')
flags.DEFINE_string('smpl_face_path', SMPL_FACE_PATH,
                    'path to smpl mesh faces (for easy rendering)')
flags.DEFINE_string('load_path', PRETRAINED_MODEL, 'path to trained model')


flags.DEFINE_integer('img_size', 224,
                     'Input image size to the network after preprocessing')
flags.DEFINE_string('data_format', 'NHWC', 'Data format')
flags.DEFINE_integer('num_stage', 3, '# of times to iterate regressor')
flags.DEFINE_boolean('ief', True, 'always true.')

# For refining:
flags.DEFINE_boolean('viz', False, 'visualize refinement')

# Weights for refinment:
flags.DEFINE_float('e_lr', 1e-3, 'step size for iteration.')
flags.DEFINE_float('e_loss_weight', 10, 'weight on kp alignment')
flags.DEFINE_float('shape_loss_weight', .5, 'weight on shape variance')
flags.DEFINE_float('joint_smooth_weight', 25, 'weight on joint smoothness')
flags.DEFINE_float('camera_smooth_weight', 1., 'weight on camera smoothness')
flags.DEFINE_float('init_pose_loss_weight', 100., 'weight on how much to stick to initial pose')

# Other settings:
flags.DEFINE_integer('num_refine', 300, 'number of iterations to optimize.')
flags.DEFINE_boolean('use_weighted_init_pose', True, 'weights init_pose_loss according to initial closeness with openpose.. ')
flags.DEFINE_boolean('refine_inpose', False, 'if true optimizes wrt the pose space as opposed to the latent feature space. ')


def get_config():
    config = flags.FLAGS
    config(sys.argv)

    setattr(config, 'img_size', 224)
    setattr(config, 'data_format', 'NHWC')

    return config
