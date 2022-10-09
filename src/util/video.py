"""
Video utils used by refine_video.
"""
from os.path import exists, join, basename
import deepdish as dd
import cv2
import numpy as np


def read_frames(path):
    vid = cv2.VideoCapture(path)

    imgs = []
    # n_frames = int(vid.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

    success = True
    while success:
        success, img = vid.read()
        if success:
            # Make BGR->RGB!!!!
            imgs.append(img[:, :, ::-1])

    vid.release()

    return imgs


def read_data(vid_path, op_dir, max_length=10000):
    """
    Returns all frames and per_frame_people
    a dict of {time, (p_id, bbox)}
    """
    valid = True
    fname = basename(vid_path).replace('.mp4', '_bboxes.h5')
    bbox_path = join(op_dir, fname)

    if not exists(bbox_path):
        print('!!!%s doesnt exist!!!' % bbox_path)
        return None, None, False
    frames = read_frames(vid_path)

    per_frame_people = dd.io.load(bbox_path)

    if len(per_frame_people.keys()) == 0:
        return None, None, False

    # Skip too long videos
    if len(frames) > max_length:
        print('Video too long!!')
        return None, None, False

    return frames, per_frame_people, valid


def openpose2cocoplus(op_kp):
    # This is what we want.
    joint_names = [
        'R Ankle', 'R Knee', 'R Hip', 'L Hip', 'L Knee', 'L Ankle', 'R Wrist',
        'R Elbow', 'R Shoulder', 'L Shoulder', 'L Elbow', 'L Wrist', 'Neck',
        'Head', 'Nose', 'L Eye', 'R Eye', 'L Ear', 'R Ear'
    ]
    # Order of open pose
    op_names = [
        'Nose',
        'Neck',
        'R Shoulder',
        'R Elbow',
        'R Wrist',
        'L Shoulder',
        'L Elbow',
        'L Wrist',
        'R Hip',
        'R Knee',
        'R Ankle',
        'L Hip',
        'L Knee',
        'L Ankle',
        'R Eye',
        'L Eye',
        'R Ear',
        'L Ear',
        'Head',
    ]

    permute_order = np.array([op_names.index(name) for name in joint_names])
    # Append a dummy 0 joint for the head.
    op_kp = np.vstack((op_kp, np.zeros((1, 3))))
    kp = op_kp[permute_order, :]

    return kp


def process_image(im, v1=False):
    """
    Pre-process normalization done to images testing.
    if v1 is True, this subtracts the mean
    else converts into [-1, 1] space.
    """
    import numpy as np
    if np.issubdtype(im.dtype, np.integer):
        # Image is [0, 255], conver to [0,1.]
        im = im / 255.
    return 2 * (im - 0.5)


def preprocess_image(frame, bbox, op_kp, img_size, vis_thresh):
    """
    Also converts op_kp into cocoplus kp.
    """
    from src.util.image import resize_img
    # bbox here is (cx, cy, scale, x, y, h, w)
    center = bbox[:2]
    scale = bbox[2]
    image_scaled, scale_factors = resize_img(frame, scale)

    # Swap so it's [x, y]
    scale_factors = [scale_factors[1], scale_factors[0]]
    center_scaled = np.round(center * scale_factors).astype(np.int)

    # re-order keypoints.
    kp = openpose2cocoplus(op_kp)
    # Scale it:
    kp[:, 0] *= scale_factors[0]
    kp[:, 1] *= scale_factors[1]

    margin = int(img_size / 2)
    image_pad = np.pad(
        image_scaled, ((margin, ), (margin, ), (0, )), mode='edge')
    center_pad = center_scaled + margin
    # figure out starting point
    start_pt = center_pad - margin
    end_pt = center_pad + margin
    # crop:
    crop = image_pad[start_pt[1]:end_pt[1], start_pt[0]:end_pt[0], :]

    kp[:, 0] -= (start_pt[0] - margin)
    kp[:, 1] -= (start_pt[1] - margin)

    kp_score = kp[:, 2]
    vis = np.expand_dims(kp_score > vis_thresh, 1)
    proc_kp = 2. * (kp * vis / img_size) - 1.
    proc_kp[:, 2] = kp_score
    # Make nonvis points 0
    proc_kp = vis * proc_kp

    # Start pt is [x, y] (after padding..) use coordinate system without padding (i.e. subtract margin)
    proc_param = {
        'scale': scale_factors,
        'start_pt': start_pt - margin,
        'target_size': img_size,
        'bbox': bbox,
        'op_kp': op_kp,
    }

    # from renderer import draw_skeleton
    # import matplotlib.pyplot as plt
    # plt.ion()
    # plt.clf()
    # fig = plt.figure(1)
    # plt.imshow(draw_skeleton(crop, kp[:, :2], vis=kp[:,2]>0.1))
    # import ipdb; ipdb.set_trace()

    proc_img = process_image(crop)
    proc_img = np.expand_dims(proc_img, 0)
    return proc_img, proc_kp, proc_param


def collect_frames(frames, per_frame_people, img_size, vis_thresh):
    time_with_people = per_frame_people.keys()
    start_frame = min(time_with_people)
    end_frame = max(time_with_people)

    # Pick one person
    use_p_id = per_frame_people[start_frame][0]

    proc_imgs, kps, proc_params = [], [], []

    for i in range(start_frame, end_frame + 1):
        frame = frames[i]
        people_here = per_frame_people[i]
        for p_id, bbox, op_kp in people_here:
            if p_id != use_p_id:
                continue
        proc_img, kp, proc_param = preprocess_image(frame, bbox, op_kp,
                                                    img_size, vis_thresh)
        proc_imgs.append(proc_img)
        kps.append(kp)
        proc_params.append(proc_param)

    return proc_imgs, kps, proc_params, start_frame, end_frame
