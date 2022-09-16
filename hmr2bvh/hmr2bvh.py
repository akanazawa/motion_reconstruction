from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from glob import glob
from os.path import join, exists
import tensorflow as tf

from bvh_core import write2bvh

kVidDir = '/Users/kanazawa/Dropbox/research/video_imitation/final_data/data/'
_SMPL_MODEL_PATH = '/Users/kanazawa/Dropbox/research/humane2e/lib/neutral_smpl_with_cocoplus_reg.pkl'

tf.app.flags.DEFINE_string('vid_dir', kVidDir, 'directory with videos')
FLAGS = tf.app.flags.FLAGS

def main(vid_dir):
    # Video with result.
    res_paths = sorted(glob(join(FLAGS.vid_dir, "*.h5")))

    for res_path in res_paths[::-1]:
        vid_path = res_path.replace('.npy', '.mp4')
        bvh_path = res_path.replace('.npy', '.bvh')


        write2bvh(res_path, bvh_path)

if __name__ == '__main__':
    main(FLAGS.vid_dir)

