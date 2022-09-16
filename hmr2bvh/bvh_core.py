import numpy as np

import deepdish as dd
import math
import cv2

from smpl_webuser.serialization import load_model

# The vertex id for the joint corresponding to the head.
_HEAD_VID = 411

_SMPL_MODEL_PATH = '/scratch1/Dropbox/research/humane2e/lib/neutral_smpl_with_cocoplus_reg.pkl'
smpl = load_model(_SMPL_MODEL_PATH)



def fill_header(beta, num_frames, fps=30):
    from bvh_template_smplspine_zyx import template

    smpl.pose[:] = 0.
    smpl.betas[:] = beta
    smpl.trans[:] = 0.

    # Construct 25 joints with head top.
    head_pt_orig = smpl.r[_HEAD_VID, :]
    smpl_joints = np.vstack((np.copy(smpl.J.r), head_pt_orig))

    """
    0: Root
    1: LHip
    2: RHip
    3: Waist (above root)
    4: LKnee
    5: RKnee
    6: Spine (above Waist)
    7: LAnkle
    8: RAnkle
    9: Chest
    10: LToe
    11: RToe
    12: Neck
    13: LInnerShould
    14: RInnerShould
    15: Chin (above throat)
    16: LShould
    17: RShould
    18: LElbow
    19: RElbow
    20: LWrist
    21: RWrist
    22: LHand
    23: RHand
    (24: Head)
    
    Spine connection:
    Root -> Waist -> Spine -> Chest -> L: {InnerShould --> Shoulde}, R:{InnerShould --> Shoulder}, Neck
    """

    # SMPL joints to use in the order:
    joint_names = [
        'Root', 'LHip', 'LKnee', 'LFoot', 'LToe', 'RHip', 'RKnee', 'RFoot',
        'RToe', 'Waist', 'Spine', 'Chest', 'Neck', 'Head', 'LInShould',
        'LShould', 'LElbow', 'LWrist', 'RInShould', 'RShould', 'RElbow',
        'RWrist'
    ]
    # Correspoinding joints.
    joint_index = [
        0, 1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12, 24, 13, 16, 18, 20, 14, 17,
        19, 21
    ]
    # Index to each parents
    parent_names = [
        'na', 'Root', 'LHip', 'LKnee', 'LFoot', 'Root', 'RHip', 'RKnee',
        'RFoot', 'Root', 'Waist', 'Spine', 'Chest', 'Neck', 'Chest',
        'LInShould', 'LShould', 'LElbow', 'Chest', 'RInShould', 'RShould',
        'RElbow'
    ]
    # kintree_table = smpl.kintree_table
    # parent = {i : kintree_table[0,i] for i in range(1, kintree_table.shape[1])}

    # np.array([(jn, pn) for jn, pn in zip(joint_names, parent_names)])
    # Relevant joints.
    joints = smpl_joints[joint_index, :]

    # Gather offsets to write to template.
    offsets = []
    for id, joint_name in enumerate(joint_names):
        if id == 0:
            offset_here = joints[id, :]
        else:
            parent_name = parent_names[id]
            parent_id = joint_names.index(parent_name)
            # offset_here = joints[parent_id,:] - joints[id,:]
            offset_here = joints[id, :] - joints[parent_id, :]

        offsets.append(offset_here)

    # Offsets should be length 22
    offsets = np.stack(offsets).ravel() * 100
    write_data = offsets.tolist() + [num_frames, 1. / fps]
    header = template.format(d=write_data)

    # print(header)
    return header


def euler2rot(theta):
    R_x = np.array([[1, 0, 0], [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]),
                     math.cos(theta[0])]])

    R_y = np.array([[math.cos(theta[1]), 0,
                     math.sin(theta[1])], [0, 1, 0],
                    [-math.sin(theta[1]), 0,
                     math.cos(theta[1])]])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]),
                     math.cos(theta[2]), 0], [0, 0, 1]])

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R


def rot2euler(R):
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def axis2euler(axis_angle):
    """
    Converts axis angle to euler angles.
    """
    R = cv2.Rodrigues(axis_angle)[0]
    euler = rot2euler(R)
    R_hat = euler2rot(euler)
    assert (np.all(R - R_hat < 1e-3))

    return euler


def euler2rot_zyx(theta):
    z, y, x = theta

    R_x = np.array([[1, 0, 0], [0, math.cos(x), -math.sin(x)],
                    [0, math.sin(x),
                     math.cos(x)]])

    R_y = np.array([[math.cos(y), 0,
                     math.sin(y)], [0, 1, 0],
                    [-math.sin(y), 0,
                     math.cos(y)]])

    R_z = np.array([[math.cos(z), -math.sin(z), 0],
                    [math.sin(z),
                     math.cos(z), 0], [0, 0, 1]])

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R


def write2bvh(res_path, out_path, DRAW=False):
    if '.h5' in res_path:
        res = dd.io.load(res_path)
    else:
        res = np.load(res_path).item()

    num_frames = len(res)
    # Assuming there is only one person..
    p_id = 0
    # With orth cam assumption.
    cam_len = 3
    # Shapes
    if 'theta' not in res[0][0].keys():
        return

    betas = np.array(
        [res[f_id][p_id]['theta'][0, cam_len + 72:] for f_id in res.keys()])
    mean_beta = np.mean(betas, axis=0)

    bvh_header = fill_header(mean_beta, num_frames)

    # SMPL joints to use in the order:
    # joint_names = ['Root', 'LHip', 'LKnee', 'LFoot', 'LToe', 'RHip', 'RKnee', 'RFoot', 'RToe', 'Waist', 'Neck', 'Head', 'LShould', 'LElbow', 'LWrist', 'RShould', 'RElbow', 'RWrist']
    # Correspoinding joints.
    # joint_index = [0, 1, 4, 7, 10, 2, 5, 8, 11, 3, 12, 24, 16, 18, 20, 17, 19, 21]

    # SMPL joints to use in the order:
    joint_names = [
        'Root', 'LHip', 'LKnee', 'LFoot', 'LToe', 'RHip', 'RKnee', 'RFoot',
        'RToe', 'Waist', 'Spine', 'Chest', 'Neck', 'Head', 'LInShould',
        'LShould', 'LElbow', 'LWrist', 'RInShould', 'RShould', 'RElbow',
        'RWrist'
    ]
    # Correspoinding joints.
    joint_index = [
        0, 1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12, 24, 13, 16, 18, 20, 14, 17,
        19, 21
    ]

    # But, 'LToe', 'RToe', 'LWrist', 'RWrist' are there just for offset.
    ignore_these = ['LToe', 'RToe', 'LWrist', 'RWrist', 'Head']
    # Joints to read off rotation from..
    rot_index = [
        joint_index[jid] for jid in range(len(joint_index))
        if joint_names[jid] not in ignore_these
    ]

    test = [(jid, joint_index[jid], joint_names[jid])
            for jid in range(len(joint_index))
            if joint_names[jid] not in ignore_these]

    # Just basic assumption to go from orth to tz
    flength = 500.

    """        
    # For debug:
    from os.path import basename, exists        
    orig_mov_path = '/scratch1/Dropbox/research/video_imitation/videos_to_run/' + basename(res_path).replace('.npy', '.mp4')
    frames = []
    vid = cv2.VideoCapture(orig_mov_path)
    success = True
    while success:
        success, img = vid.read()
        if success:
            # Make BGR->RGB!!!!
            frames.append(img[:, :, ::-1])
    vid.release()
    """

    motion_strs = []
    for f_id in res.keys():
        pred = res[f_id][p_id]
        # Pose
        theta = pred['theta'][0, cam_len:cam_len + 72]
        theta = theta.reshape(-1, 3)

        cams = pred['theta'][0, :cam_len]

        # Get rotation (13 x 3)
        write_rots = theta[rot_index, :]

        # Global
        if False:
            # Don't multiply.
            global_rot_axisa = theta[0]
        else:
            coord_rotation = -np.eye(3)
            coord_rotation[0, 0] = 1.
            # Global needs to be right multiplied by this..
            global_rot_axisa = cv2.Rodrigues(
                cv2.Rodrigues(theta[0])[0].dot(coord_rotation))[0]
        global_rot = np.rad2deg(axis2euler(global_rot_axisa))
        global_rot = global_rot[::-1]

        # In euler angles.
        write_rots_euler = np.array(
            [axis2euler(axisangle) for axisangle in write_rots])
        relative_rot = np.rad2deg(write_rots_euler[1:])
        relative_rot = relative_rot[:, ::-1]

        target_size = pred['proc_param']['target_size']
        tz = flength / (0.5 * target_size * cams[0])

        # joints2D in image coord is..
        # ((sX + [tx;ty]) + 1 + 0.5 * img_size + start_pt) * scale
        # txty = (np.array([cams[1], cams[2]]) + 1) * 0.5 * target_size
        # This is in original image coord..
        undo_scale = 1. / np.array(pred['proc_param']['scale'])
        # Add the bbox center!
        # txty1 = (txty - pred['proc_param']['start_pt']) * undo_scale
        # txty1 = (txty + pred['proc_param']['start_pt']) * undo_scale
        # Need to compute global tx, ty, tz.
        # global_trans = np.hstack([txty1, tz])

        left_hip, right_hip = 3, 2
        joints = pred['joints'][0]
        # this is in cropped
        pelvis = (joints[left_hip] + joints[right_hip]) / 2.
        # in original image coordinate
        pelvis_img = ((
            (pelvis + 1) * 0.5 * target_size) + pred['proc_param']['start_pt']
                      ) * undo_scale
        global_trans = np.hstack([pelvis_img, tz])

        # Vis:
        # cx, cy, bsc = pred['bbox'][:3]
        # center = np.array([cx, cy])
        # from datasets.common import resize_img
        # image_scaled, scale_factors = resize_img(img, bsc)
        # # Swap so it's [x, y]
        # scale_factors = [scale_factors[1], scale_factors[0]]
        # center_scaled = center * scale_factors
        # margin = int(target_size/2)
        # # These are all equal:
        # start_pt_img = (center_scaled - margin) * 1./bsc
        # top_corner = center - (0.5 * target_size * 1./bsc)

        ## Viz:
        # img = frames[f_id]
        # bbox = pred['bbox'][-4:]
        # x,y = (bbox[:2]).astype(np.int)
        # width,height = (bbox[2:]).astype(np.int)
        # cv2.rectangle(img[:, :, ::-1], (x, y), (x+width, y+height), (158, 202, 225), 2)
        # import matplotlib.pyplot as plt
        # plt.ion()
        # plt.clf()
        # plt.imshow(img)
        # op_kp = pred['op_kp'][:, :2]
        # pred_kp = (pred['joints'][0] + 1) * 0.5 * target_size
        # pred_kp = (pred_kp + pred['proc_param']['start_pt']) * undo_scale
        # (pred_kp[left_hip] + pred_kp[right_hip]) / 2
        # # plt.scatter(op_kp[:, 0], op_kp[:, 1])
        # plt.scatter(pred_kp[:, 0], pred_kp[:, 1])
        # plt.scatter(pelvis_img[0], pelvis_img[1])
        # import ipdb; ipdb.set_trace()

        motion = np.vstack((global_trans, global_rot, relative_rot)).ravel()

        motion_strs.append(' '.join(['%.5f' % m for m in motion]))

        if DRAW:
            # Visualize
            # smpl.trans[:] = global_trans
            smpl.pose[:] = theta.ravel()

            import ipdb
            ipdb.set_trace()

    # Make motion_strs into a big str
    motion_str = '\n'.join(motion_strs)

    print('writing to %s' % out_path)
    with open(out_path, "w") as f:
        f.write(bvh_header)
        f.write(motion_str)


def read_bvh(bvh_path):
    """
    Reads subset of HMR from jason.
    """
    f = file(bvh_path).read()
    words = f.split()
    addNext = 0

    boneName = []
    frameCount = 0

    for word in words:
        if addNext > 0:
            if addNext == 6:
                boneName.append(word + 'Xpos')
                boneName.append(word + 'Ypos')
                boneName.append(word + 'Zpos')
                boneName.append(word + 'Yrot')
                boneName.append(word + 'Xrot')
                boneName.append(word + 'Zrot')
            else:
                boneName.append(word + 'Yrot')
                boneName.append(word + 'Xrot')
                boneName.append(word + 'Zrot')

            addNext = 0

        if word == 'ROOT':
            addNext = 6
        if word == 'JOINT':
            addNext = 3

        if word == 'Frames:':
            frameCount = int(words[words.index('Frames:') + 1])

        if word == 'Time:':
            content = words[words.index('Time:') + 2:]
            break

    data = [[] for _ in range(len(boneName))]
    frames = []
    size = len(boneName)

    # This is #frames x 54
    lines = np.array([float(cont) for cont in content]).reshape(-1, size)
    # Make 600 fps -> 30fps
    lines = lines[::20]

    def convert2smpl(entry):
        # converts 54D of euler to smpl params.
        trans = entry[:3]
        # 17 x 3
        eulers = np.deg2rad(entry[3:].reshape(-1, 3))
        Rs_orig = [euler2rot_zyx(euler) for euler in eulers]
        Rs_orig[13] = euler2rot_zyx(-eulers[13][[1, 0, 2]])
        Rs_orig[16] = euler2rot_zyx(eulers[16][[1, 0, 2]])

        aroundy = cv2.Rodrigues(np.array([0, -np.pi/2, 0]))[0]
        Rs = [aroundy.dot(R).dot(aroundy.T) for R in Rs_orig]

        aroundypi = cv2.Rodrigues(np.array([0, np.pi, 0]))[0]
        Rs[0] = aroundypi.dot(Rs[0])

        # Tried:
        trans = (cv2.Rodrigues(np.array([0, np.pi/2, 0]))[0]).dot(trans)

        # This is 17x3 (root + 16 joints)
        aangles = np.array([cv2.Rodrigues(R)[0].ravel() for R in Rs])

        # Order is:
        # SMPL joints to use in the order:
        all_smpl_names = [
            'Root',
            'LHip',
            'RHip',
            'Waist',
            'LKnee',
            'RKnee',
            'Spine',
            'LFoot',
            'RFoot',
            'Chest',
            'LToe',
            'RToe',
            'Neck',
            'LInnerShould',
            'RInnerShould',
            'Chin',
            'LShould',
            'RShould',
            'LElbow',
            'RElbow',
            'LWrist',
            'RWrist',
            'LHand',
            'RHand',
        ]
        # Incoming:
        joint_names = [
            'Root', 'LHip', 'LKnee', 'LFoot', 'RHip', 'RKnee', 'RFoot',
            'Waist', 'Spine', 'Chest', 'Neck', 'LInnerShould', 'LShould',
            'LElbow', 'RInnerShould', 'RShould', 'RElbow',
        ]
        # joint_names = [
        #     'Root', 'RHip', 'RKnee', 'RFoot', 'LHip', 'LKnee', 'LFoot',
        #     'Waist', 'Spine', 'Chest', 'Neck', 'RInnerShould', 'RShould',
        #     'RElbow', 'LInnerShould', 'LShould', 'LElbow',
        # ]
        # joint_names = [
        #     'Root', 'LHip', 'LKnee', 'LFoot', 'RHip', 'RKnee', 'RFoot',
        #     'Waist', 'Spine', 'Chest', 'Neck', 'LInnerShould', 'LShould',
        #     'RElbow', 'RInnerShould', 'RShould', 'LElbow',
        # ]
        
        # Incoming index.
        joint_index = [all_smpl_names.index(jn) for jn in joint_names]

        pose = np.zeros((len(all_smpl_names), 3))

        pose[joint_index, :] = aangles

        return (pose, trans)
        import ipdb
        ipdb.set_trace()

    smpl_params = [convert2smpl(line) for line in lines]

    poses = [sp[0] for sp in smpl_params]
    trans = [sp[1] for sp in smpl_params]

    return poses, trans
