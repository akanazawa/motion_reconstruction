"""
Similar to RunModel, but fine-tunes over time on openpose output.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .ops import keypoint_l1_loss
from .models import Encoder_resnet, Encoder_fc3_dropout
from .tf_smpl.batch_lbs import batch_rodrigues
from .tf_smpl.batch_smpl import SMPL
from .tf_smpl.projection import batch_orth_proj_idrot
from .util.renderer import SMPLRenderer, draw_skeleton
from .util.image import unprocess_image
import time
from os.path import exists

import tensorflow as tf
import numpy as np


class Refiner(object):
    def __init__(self, config, num_frames, sess=None):
        """
        Args:
          config,,
        """
        # Config + path
        if not config.load_path:
            raise Exception(
                "[!] You should specify `load_path` to load a pretrained model")
        if not exists(config.load_path + '.index'):
            print('%s doesnt exist..' % config.load_path)
            import ipdb
            ipdb.set_trace()
        self.config = config
        self.load_path = config.load_path
        self.num_frames = num_frames

        self.data_format = config.data_format
        self.smpl_model_path = config.smpl_model_path

        # Visualization for fitting
        self.viz = config.viz
        self.viz_sub = 10

        # Loss & Loss weights:
        self.e_lr = config.e_lr

        self.e_loss_weight = config.e_loss_weight
        self.shape_loss_weight = config.shape_loss_weight
        self.joint_smooth_weight = config.joint_smooth_weight
        self.camera_smooth_weight = config.camera_smooth_weight
        self.keypoint_loss = keypoint_l1_loss
        self.init_pose_loss_weight = config.init_pose_loss_weight

        # Data
        self.batch_size = num_frames
        self.img_size = config.img_size

        input_size = (self.batch_size, self.img_size, self.img_size, 3)
        self.images_pl = tf.placeholder(tf.float32, shape=input_size)
        self.img_feat_pl = tf.placeholder(tf.float32, shape=(self.batch_size, 2048))
        self.img_feat_var = tf.get_variable("img_feat_var", dtype=tf.float32, shape=(self.batch_size, 2048))        
        kp_size = (self.batch_size, 19, 3)
        self.kps_pl = tf.placeholder(tf.float32, shape=kp_size)

        # Camera type!
        self.num_cam = 3
        self.proj_fn = batch_orth_proj_idrot
        self.num_theta = 72  # 24 * 3
        self.total_params = self.num_theta + self.num_cam + 10

        # Model spec
        # For visualization
        if self.viz:
            self.renderer = SMPLRenderer(img_size=self.img_size, face_path=config.smpl_face_path)

        # Instantiate SMPL
        self.smpl = SMPL(self.smpl_model_path)

        self.theta0_pl_shape = [self.batch_size, self.total_params]
        self.theta0_pl = tf.placeholder_with_default(
            self.load_mean_param(), shape=self.theta0_pl_shape, name='theta0')

        # Optimization space.
        self.refine_inpose = config.refine_inpose
        if self.refine_inpose:
            self.theta_pl = tf.placeholder(tf.float32, shape=self.theta0_pl_shape, name='theta_pl')
            self.theta_var = tf.get_variable("theta_var", dtype=tf.float32, shape=self.theta0_pl_shape)

        # For ft-loss
        self.shape_pl = tf.placeholder_with_default(tf.zeros(10), shape=(10,), name='beta0')
        # For stick-to-init-pose loss:
        self.init_pose_pl = tf.placeholder_with_default(tf.zeros([num_frames, 72]), shape=(num_frames, 72), name='pose0')
        self.init_pose_weight_pl = tf.placeholder_with_default(tf.ones([num_frames, 1]), shape=(num_frames, 1), name='pose0_weights')
        # For camera loss
        self.scale_factors_pl = tf.placeholder_with_default(tf.ones([num_frames]), shape=(num_frames), name='scale_factors')
        self.offsets_pl = tf.placeholder_with_default(tf.zeros([num_frames, 2]), shape=(num_frames, 2), name='offsets')

        # Build model!
        self.ief = config.ief
        if self.ief:
            self.num_stage = config.num_stage
            self.build_refine_model()
        else:
            print('never here')
            import ipdb
            ipdb.set_trace()
            
        all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        # Exclude the new variable.
        all_vars_filtered = [v for v in all_vars if ('img_feat_var' not in v.name) and ('theta_var' not in v.name)]
        self.saver = tf.train.Saver(all_vars_filtered)

        if sess is None:
            self.sess = tf.Session()            
        else:
            self.sess = sess

        new_vars = [v for v in all_vars if ('img_feat_var' in v.name) or ('theta_var' in v.name)]
        self.sess.run(tf.variables_initializer(new_vars))

        # coord = tf.train.Coordinator()
        # threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
        self.prepare()

    def load_mean_param(self):
        mean = np.zeros((1, self.total_params))
        mean[0, 0] = 0.9  # This is scale.

        mean = tf.constant(mean, tf.float32)

        self.mean_var = tf.Variable(
            mean, name="mean_param", dtype=tf.float32, trainable=True)
        # self.E_var.append(self.mean_var)
        init_mean = tf.tile(self.mean_var, [self.batch_size, 1])
        # 85D consists of [cam (3), pose (72), shapes (10)]
        # cam is [scale, tx, ty]
        return init_mean

    def prepare(self):
        print('Restoring checkpoint %s..' % self.load_path)
        self.saver.restore(self.sess, self.load_path)
        self.mean_value = self.sess.run(self.mean_var)

    def build_refine_model(self):
        img_enc_fn = Encoder_resnet
        threed_enc_fn = Encoder_fc3_dropout
        
        self.img_feat, self.E_var = img_enc_fn(
            self.images_pl,
            is_training=False,
            reuse=False)

        self.set_img_feat_var = self.img_feat_var.assign(self.img_feat_pl)

        # Start loop
        self.all_verts = []
        self.all_kps = []
        self.all_cams = []
        self.all_Js = []
        self.all_Jsmpl = []
        self.final_thetas = []
        theta_prev = self.theta0_pl
        for i in np.arange(self.num_stage):
            print('Iteration %d' % i)
            # ---- Compute outputs
            state = tf.concat([self.img_feat_var, theta_prev], 1)

            if i == 0:
                delta_theta, threeD_var = threed_enc_fn(
                    state,
                    num_output=self.total_params,
                    is_training=False,
                    reuse=False)
                self.E_var.append(threeD_var)
            else:
                delta_theta, _ = threed_enc_fn(
                    state,
                    num_output=self.total_params,
                    is_training=False,
                    reuse=True)
            # Compute new theta
            theta_here = theta_prev + delta_theta
            # cam = N x 3, pose N x self.num_theta, shape: N x 10
            cams = theta_here[:, :self.num_cam]
            poses = theta_here[:, self.num_cam:(self.num_cam + self.num_theta)]
            shapes = theta_here[:, (self.num_cam + self.num_theta):]
            # Rs_wglobal is Nx24x3x3 rotation matrices of poses
            verts, Js, pred_Rs = self.smpl(shapes, poses, get_skin=True)
            Jsmpl = self.smpl.J_transformed

            # Project to 2D!
            pred_kp = self.proj_fn(Js, cams, name='proj_2d_stage%d' % i)
            self.all_verts.append(verts)
            self.all_kps.append(pred_kp)
            self.all_cams.append(cams)
            self.all_Js.append(Js)
            self.all_Jsmpl.append(Jsmpl)
            # save each theta.
            self.final_thetas.append(theta_here)
            # Finally)update to end iteration.
            theta_prev = theta_here

        # Compute everything with the final theta.
        if self.refine_inpose:
            self.set_theta_var = self.theta_var.assign(self.theta_pl)
            theta_final = self.theta_var
        else:
            theta_final = theta_here

        cams = theta_final[:, :self.num_cam]
        poses = theta_final[:, self.num_cam:(self.num_cam + self.num_theta)]
        shapes = theta_final[:, (self.num_cam + self.num_theta):]
        # Rs_wglobal is Nx24x3x3 rotation matrices of poses
        verts, Js, pred_Rs = self.smpl(shapes, poses, get_skin=True)
        Jsmpl = self.smpl.J_transformed
        # Project to 2D!
        pred_kp = self.proj_fn(Js, cams, name='proj_2d_stage%d' % i)

        self.all_verts.append(verts)
        self.all_kps.append(pred_kp)
        self.all_cams.append(cams)
        self.all_Js.append(Js)
        self.all_Jsmpl.append(Jsmpl)
        # save each theta.
        self.final_thetas.append(theta_final)

        # Compute new losses!!
        self.e_loss_kp = self.e_loss_weight * self.keypoint_loss(self.kps_pl,
                                                                 pred_kp)
        # Beta variance should be low!
        self.loss_shape = self.shape_loss_weight * shape_variance(shapes, self.shape_pl)
        self.loss_init_pose = self.init_pose_loss_weight * init_pose(pred_Rs, self.init_pose_pl, weights=self.init_pose_weight_pl)
        # Endpoints should be smooth!!
        self.loss_joints = self.joint_smooth_weight * joint_smoothness(Js)
        # Camera should be smooth
        self.loss_camera = self.camera_smooth_weight * camera_smoothness(cams, self.scale_factors_pl, self.offsets_pl, img_size=self.config.img_size)

        self.total_loss = self.e_loss_kp + self.loss_shape + self.loss_joints + self.loss_init_pose + self.loss_camera

        # Setup optimizer
        print('Setting up optimizer..')
        self.optimizer = tf.train.AdamOptimizer
        e_optimizer = self.optimizer(self.e_lr)

        if self.refine_inpose:
            self.e_opt = e_optimizer.minimize(self.total_loss, var_list=[self.theta_var])
        else:
            self.e_opt = e_optimizer.minimize(self.total_loss, var_list=[self.img_feat_var])

        print('Done initializing the model!')


    def predict(self, images, kps, scale_factors, offsets):
        """
        images: num_batch, img_size, img_size, 3
        kps: num_batch x 19 x 3
        Preprocessed to range [-1, 1]

        scale_factors, offsets: used to preprocess the bbox

        Runs the model with images.
        """
        ## Initially, get the encoding of images:
        feed_dict = { self.images_pl: images }
        fetch_dict = {'img_feats': self.img_feat}

        img_feats = self.sess.run(self.img_feat, feed_dict)

        feed_dict = {
            self.img_feat_pl: img_feats,
            self.kps_pl: kps,
        }

        self.sess.run(self.set_img_feat_var, feed_dict)

        if self.refine_inpose:
            # Take -2 bc that's the actual theta (-1 is still placeholder)
            use_res = -2
        else:
            use_res = -1
        fetch_dict = {
            'theta': self.final_thetas[use_res],
            'joints': self.all_kps[use_res],
            'verts': self.all_verts[use_res],
        }

        init_result = self.sess.run(fetch_dict, feed_dict)

        shapes = init_result['theta'][:, -10:]
        # Save mean shape of this trajectory:
        mean_shape = np.mean(shapes, axis=0)
        feed_dict[self.shape_pl] = mean_shape

        # Save initial pose output:
        init_pose = init_result['theta'][:, 3:3+72]
        feed_dict[self.init_pose_pl] = init_pose
        # import ipdb; ipdb.set_trace()

        if self.refine_inpose:
            print('Doing optimization in pose space!!')
            feed_dict[self.theta_pl] = init_result['theta']
            self.sess.run(self.set_theta_var, feed_dict)

        if self.config.use_weighted_init_pose:
            print('Using weighted init pose!')
            # Compute weights according to op match.
            gt_kps = np.stack(kps)
            vis = gt_kps[:, :, 2, None]
            diff = vis * (gt_kps[:, :, :2] - init_result['joints'])
            # L2:
            error = np.linalg.norm(diff.reshape(diff.shape[0], -1), axis=1, ord=1)
            # N x 1
            weights = np.expand_dims(np.exp(-(error / error.max())**2), 1)

            feed_dict[self.init_pose_weight_pl] = weights

        # Add scale factor & offsets for camera loss.
        feed_dict[self.scale_factors_pl] = scale_factors
        feed_dict[self.offsets_pl] = offsets

        # All fetches:
        fetch_dict = {
            'joints': self.all_kps[-1],
            'verts': self.all_verts[-1],
            'cams': self.all_cams[-1],
            'joints3d': self.all_Js[-1],
            'theta': self.final_thetas[-1],
            'total_loss': self.total_loss,
            'loss_kp': self.e_loss_kp,
            'loss_shape': self.loss_shape,
            'loss_joints': self.loss_joints,
            'loss_camera': self.loss_camera,
            'loss_init_pose': self.loss_init_pose,
            'optim': self.e_opt,
        }

        if self.viz:
            # Precompute op imgs bc they are constant
            crops = [
                unprocess_image(image) for image in images[::self.viz_sub]
            ]
            op_kps = [
                np.hstack((self.img_size * ((kp[:, :2] + 1) * 0.5), kp[:, 2, None]))
                for kp in kps[::self.viz_sub]
            ]
            op_imgs = [
                draw_skeleton(crop, kp, vis=kp[:, 2] > 0.1)
                for crop, kp in zip(crops, op_kps)
            ]
            op_img = np.hstack(op_imgs)

            proj_img, rend_img = self.visualize_seq(crops, init_result)
            import matplotlib.pyplot as plt
            plt.ion()
            plt.figure(1)
            plt.clf()
            plt.suptitle('init')
            plt.subplot(311)
            plt.imshow(op_img)
            plt.axis('off')
            plt.title('openpose joints')
            plt.subplot(312)
            plt.imshow(proj_img)
            plt.axis('off')
            plt.title('proj joints')
            plt.subplot(313)
            plt.imshow(rend_img)
            plt.axis('off')
            plt.title('rend verts')
            plt.pause(1e-3)
            import ipdb; ipdb.set_trace()

        all_loss_keys = ['loss_kp', 'loss_shape', 'loss_joints', 'loss_init_pose', 'loss_camera']
        tbegin = time.time()
        num_iter = self.config.num_refine
        loss_records = {}
        for step in xrange(num_iter):
            result = self.sess.run(fetch_dict, feed_dict)
            loss_keys = [key for key in all_loss_keys if key in result.keys()]
            total_loss = result['total_loss']
            msg_prefix = 'iter %d/%d, total_loss %.2g' % (step, num_iter, total_loss)
            msg_raw = ['%s: %.2g' % (key, result[key]) for key in loss_keys]
            print(msg_prefix + ' ' + ' ,'.join(msg_raw))

            if step == 0:
                for key in loss_keys:
                    loss_records[key] = [result[key]]
            else:
                for key in loss_keys:
                    loss_records[key].append(result[key])

            # Do visualization
            if self.viz and step > 0 and step % 50 == 0:
                proj_img, rend_img = self.visualize_seq(crops, result)
                import matplotlib.pyplot as plt
                plt.ion()
                plt.figure(2)
                plt.clf()                
                plt.suptitle('iter %d' % step) 
                plt.subplot(311)
                plt.imshow(op_img)
                plt.axis('off')
                plt.title('openpose joints')
                plt.subplot(312)
                plt.imshow(proj_img)
                plt.axis('off')                
                plt.title('proj joints')
                plt.subplot(313)
                plt.imshow(rend_img)
                plt.axis('off')                                
                plt.title('rend verts')
                plt.pause(1e-3)
                import ipdb; ipdb.set_trace()                

        total_time = time.time() - tbegin
        print('Total time %g' % total_time)

        del result['optim']
        del result['verts']

        result['loss_records'] = loss_records

        return result

    def visualize_seq(self, crops, result):
        """
        For weight tuning,, see:
        first row, the original renders (or original kp)
        second row, open pose
        third row, proj kp
        For every 10th frame or something like this.
        """
        pred_kps = (result['joints'][::self.viz_sub] + 1) * 0.5 * self.img_size

        proj_imgs = [
            draw_skeleton(crop, kp) for crop, kp in zip(crops, pred_kps)
        ]
        proj_img = np.hstack(proj_imgs)

        t0 = time.time()
        # Use camera to figure out the z.
        cam_scales = result['theta'][:, :0]
        tzs = [500. / (0.5 * self.config.img_size * cam_s) for cam_s in cam_scales]
        # rend_imgs = [
        #     self.renderer(vert + np.array([0, 0, tz])) for (vert, tz) in zip(result['verts'][::self.viz_sub], tzs[::self.viz_sub])
        # ]
        rend_imgs = [
            self.renderer(vert + np.array([0, 0, 6])) for vert in result['verts'][::self.viz_sub]
        ]        
        rend_img = np.hstack(rend_imgs)
        t1 = time.time()
        print('Took %f sec to render %d imgs' % (t1 - t0, len(rend_imgs)))

        return proj_img, rend_img

# All the  loss functions.

def shape_variance(shapes, target_shape=None):
    # Shapes is F x 10
    # Compute variance.
    if target_shape is not None:
        N = tf.shape(shapes)[0]
        target_shapes = tf.tile(tf.expand_dims(target_shape, 0), [N, 1])
        return tf.losses.mean_squared_error(target_shapes, shapes)
    else:
        _, var = tf.nn.moments(shapes, axes=0)
        return tf.reduce_mean(var)


def pose_smoothness(poses, global_only=False):
    """
    # Poses is F x 24 x 3 x 3
    Computes \sum ||p_i - p_{i+1}||
    On the pose in Rotation matrices space.
    It compues the angle between the two rotations:
    (tr(R) - 1) / 2 = cos(theta)
    So penalize acos((tr(R) - 1) / 2) --> this nans
    So:
    minimize: (1 - tr(R_1*R_2')) / 2 = -cos(theta) of R_1*R_2'
    min at -1.
    """
    # These are F-1 x 24 x 3 x 3 (Ok this is exactly the same..)
    curr_pose = poses[:-1]
    next_pose = poses[1:]
    RRt = tf.matmul(curr_pose, next_pose, transpose_b=True)

    # For min (1-tr(RR_T)) / 2
    costheta = (tf.trace(RRt) - 1) / 2.
    target = tf.ones_like(costheta)
    if global_only:
        print('Pose smoothness increased on global!')
        weights_global = 10 * tf.expand_dims(tf.ones_like(costheta[:, 0]), 1)
        weights_joints = tf.ones_like(costheta[:, 1:])
        weights = tf.concat([weights_global, weights_joints], 1)
    else:
        weights = tf.ones_like(costheta)
    return tf.losses.mean_squared_error(target, costheta, weights=weights)


def joint_smoothness(joints):
    """
    joints: N x 19 x 3
    Computes smoothness of joints relative to root.
    """
    if joints.shape[1] == 19:
        left_hip, right_hip = 3, 2
        root = (joints[:, left_hip] + joints[:, right_hip]) / 2.
        root = tf.expand_dims(root, 1)

        joints = joints - root
    else:
        print('Unknown skeleton type')
        import ipdb; ipdb.set_trace()

    curr_joints = joints[:-1]
    next_joints = joints[1:]
    return tf.losses.mean_squared_error(curr_joints, next_joints)


def init_pose(pred_Rs, init_pose, weights=None):
    """
    Should stay close to initial weights
    pred_Rs is N x 24 x 3 x 3
    init_pose is 72D, need to conver to Rodrigues
    """
    init_Rs = batch_rodrigues(tf.reshape(init_pose, [-1, 3]))
    init_Rs = tf.reshape(init_Rs, [-1, 24, 3, 3])
    RRt = tf.matmul(init_Rs, pred_Rs, transpose_b=True)
    costheta = (tf.trace(RRt) - 1) / 2.
    target = tf.ones_like(costheta)
    if weights is None:
        weights = tf.ones_like(costheta)
    return tf.losses.mean_squared_error(target, costheta, weights=weights)


def camera_smoothness(cams, scale_factors, offsets, img_size=224):
    # cams: [s, tx, ty]

    scales = cams[:, 0]
    actual_scales = scales * (1./scale_factors)
    trans = cams[:, 1:]
    # pred trans + bbox top left corner / img_size
    actual_trans = ((trans + 1) * img_size * 0.5 + offsets) / img_size

    curr_scales = actual_scales[:-1]
    next_scales = actual_scales[1:]

    curr_trans = actual_trans[:-1]
    next_trans = actual_trans[1:]

    scale_diff = tf.losses.mean_squared_error(curr_scales, next_scales)
    trans_diff = tf.losses.mean_squared_error(curr_trans, next_trans)
    return scale_diff + trans_diff
