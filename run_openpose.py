"""
Run openpose on a directory with videos.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, exists, basename
from os import makedirs, system
from glob import glob
import json
import numpy as np
import cv2
import matplotlib.patches as patches
import scipy.signal as signal
import deepdish as dd

import tensorflow as tf

from src.util.renderer import draw_openpose_skeleton

kVidDir = '/home/kanazawa/projects/hmr_sfv/demo_data/videos'
kOutDir = '/home/kanazawa/projects/hmr_sfv/demo_data/openpose_output'

kOpenPose = '/scratch1/storage/git_repos/openpose'
kOpenPoseModel = '/scratch1/storage/git_repos/Realtime_Multi-Person_Pose_Estimation/aj_finetuned_models_170k/'

tf.app.flags.DEFINE_string('video_dir', kVidDir, 'dir of vids')
tf.app.flags.DEFINE_string('out_dir', kOutDir, 'dir of output')
tf.app.flags.DEFINE_string('openpose_dir', kOpenPose, 'dir of openpose')
tf.app.flags.DEFINE_string('op_model_dir', kOpenPoseModel, 'dir of openpose model')

# Threshold for visible points
VIS_THR = 0.1
# KP is only accecptable if this many points are visible
NUM_VIS_THR = 5
# After smoothing, cut back until final conf is above this.
END_BOX_CONF = 0.1
# Required IOU to be a match
IOU_THR = 0.05
# If person hasn't appeared for this many frames, drop it.
OCCL_THR = 30
# Bbox traj must be longer than 50% of duration (duration -> max length any body was seen)
FREQ_THR = .1 #.3
# If median bbox area is less than this% of image area, kill it.
SIZE_THR = .23
# If avg score of the trajectory is < than this, kill it.
SCORE_THR = .4
# Nonmaxsupp overlap threshold
NMS_THR = 0.5

BOX_SIZE = 224
RADIUS = BOX_SIZE / 2.

FLAGS = tf.app.flags.FLAGS


def main(unused_argv):
    vid_dir = FLAGS.video_dir
    out_dir = FLAGS.out_dir
    openpose_dir = FLAGS.openpose_dir

    if FLAGS.op_model_dir != kOpenPoseModel:
        out_dir += "_nondefaultop"

    if not exists(vid_dir):
        print('%s doesnt exist' % vid_dir)
        import ipdb
        ipdb.set_trace()
    if not exists(out_dir):
        print('Making %s' % out_dir)
        makedirs(out_dir)

    vid_paths = sorted(glob(join(vid_dir, "*.mp4")))

    # cmd_base = '%s/build/examples/openpose/openpose.bin --video %%s --write_keypoint_json %%s --no_display --render_pose 1' % (
    #     openpose_dir)
    # Maximum accuracy configuration:
    cmd_base = '%s/build/examples/openpose/openpose.bin --video %%s --write_keypoint_json %%s --net_resolution "1312x736" --scale_number 4 --scale_gap 0.25 --write_images %%s --write_images_format jpg' % (
        openpose_dir)

    cmd_base += ' --model_folder %s' % FLAGS.op_model_dir    

    cmd_extra = ' --net_resolution "1312x736" --scale_number 4 --scale_gap 0.25'

    for i, vid_path in enumerate(vid_paths[::-1]):
        vid_name = basename(vid_path)[:-4]
        out_here = join(out_dir, vid_name)
        # bbox_path = join(out_dir, vid_name + '_bboxes_tmpwind25.h5')
        bbox_path = join(out_dir, vid_name + '_bboxes.h5')
        if exists(bbox_path):
            continue

        if not exists(out_here):
            print('Working on %s %d/%d' % (vid_name, i, len(vid_paths)))
            makedirs(out_here)
        if len(glob(join(out_here, "*.json"))) > 0:
            if not exists(bbox_path):
                digest_openpose(out_here, vid_path, bbox_path)
        # else:
        if not exists(bbox_path):
            cmd = cmd_base % (vid_path, out_here, out_here)
            print(cmd)
            res = system(cmd)
            if res > 0:
                print('somethign wrong?')
                import ipdb
                ipdb.set_trace()
            # print(cmd + cmd_extra)
            digest_openpose(out_here, vid_path, bbox_path)


def digest_openpose(json_dir, vid_path, bbox_path):
    print('reading %s' % vid_path)
    # Opens json, smoothes the output
    json_paths = sorted(glob(join(json_dir, "*.json")))

    all_kps = []
    for i, json_path in enumerate(json_paths):
        kps = read_json(json_path)
        all_kps.append(kps)

    # per_frame_people = clean_detections(all_kps, vid_path, vis=True)
    per_frame_people = clean_detections(all_kps, vid_path, vis=False)
    # Save to bbox_path.

    dd.io.save(bbox_path, per_frame_people)


def clean_detections(all_kps, vid_path, vis=False):
    """
    Takes keypoints and computes bboxes.
    Assigns identity to each box.
    Removes supurious boxes.
    Smoothes the boxes over time.
    """
    persons = {}
    bboxes = []
    if vis:
        frames = read_frames(vid_path)
    start_frame, end_frame = -1, -1
    for i, kps in enumerate(all_kps):
        if i % 50 == 0:
            print('%d/%d' % (i, len(all_kps)))
        if len(kps) == 0:
            continue

        bboxes = []
        valid_kps = []
        for kp in kps:
            bbox, kp_here = get_bbox(kp)
            if bbox is not None:
                bboxes.append(bbox)
                valid_kps.append(kp_here)

        if len(bboxes) == 0:
            # None of them were good.
            continue

        bboxes = np.vstack(bboxes)
        valid_kps = np.stack(valid_kps)

        bboxes, valid_kps = nonmaxsupp(bboxes, valid_kps)

        if len(persons.keys()) == 0:
            start_frame = i
            # In the beginning, add everybody.
            for j, (bbox, valid_kp) in enumerate(zip(bboxes, valid_kps)):
                persons[j] = [(i, bbox, valid_kp)]
        else:
            # Update this
            end_frame = i
            # Find matching persons.
            iou_scores = []
            for p_id, p_bboxes in persons.iteritems():
                last_time, last_bbox, last_kp = p_bboxes[-1]
                if (i - last_time) > OCCL_THR:
                    ious = -np.ones(len(bboxes))
                else:
                    ious = compute_iou(last_bbox, bboxes)
                iou_scores.append(ious)
            # num_person x bboxes_here
            iou_scores = np.vstack(iou_scores)
            num_bboxes = len(bboxes)

            num_persons = len(persons.keys())
            box_is_matched = np.zeros(num_bboxes)
            box_is_visited = np.zeros(num_bboxes)
            pid_is_matched = np.zeros(num_persons)
            counter = 0
            iou_scores_copy = np.copy(iou_scores)
            while not np.all(pid_is_matched) and not np.all(
                    box_is_visited) and not np.all(iou_scores == -1):
                row, col = np.unravel_index(iou_scores.argmax(), (num_persons,
                                                                  num_bboxes))
                box_is_visited[col] = True

                # Add this bbox to this person if enough overlap.
                if iou_scores[row,
                              col] > IOU_THR and not pid_is_matched[row] and not box_is_matched[col]:
                    persons[row].append((i, bboxes[col], valid_kps[col]))
                    pid_is_matched[row] = True
                    box_is_matched[col] = True
                # elif iou_scores[row,col] > IOU_THR:
                #     print('why here')
                #     import ipdb; ipdb.set_trace()
                # Reset this.
                iou_scores[row, :] = -1.
                counter += 1
                if counter > 100:
                    print('inflooo')
                    import ipdb
                    ipdb.set_trace()
            unmatched_boxes = bboxes[np.logical_not(box_is_matched)]
            unmatched_kps = valid_kps[np.logical_not(box_is_matched)]
            for new_j, (bbox, kp_here) in enumerate(zip(unmatched_boxes, unmatched_kps)):
                persons[num_persons + new_j] = [(i, bbox, kp_here)]

        if vis and i % 20 == 0:
            import matplotlib.pyplot as plt
            plt.ion()
            plt.clf()
            fig = plt.figure(1)
            ax = fig.add_subplot(111)
            frame = frames[i]
            ax.imshow(frame)
            ax.set_title('frame %d' % i)
            for p_id, p_bboxes in persons.iteritems():
                last_time, last_bbox, last_kps = p_bboxes[-1]
                # If was found in current frame
                if last_time == i:
                    rect = get_rect(last_bbox)
                    ax.add_patch(rect)
                    plt.text(last_bbox[4], last_bbox[5], 'pid: %d' % p_id)
            plt.pause(1e-3)

    # Now clean these people!
    if not vis:
        frames = read_frames(vid_path, 1)
    img_area = frames[0].shape[0] * frames[0].shape[1]
    duration = float(end_frame - start_frame)
    # orig_persons = persons.copy()
    for p_id in persons.keys():
        med_score = np.median([bbox[3] for (_, bbox, _) in persons[p_id]])
        freq = len(persons[p_id]) / duration
        median_bbox_area = np.median(
            [bbox[6] * bbox[7] for (_, bbox, _) in persons[p_id]]) / float(img_area)
        # print('freq %.2f, score %.2f, size %.2f' % (freq, med_score, median_bbox_area))
        if freq < FREQ_THR:
            print('Rejecting %d bc too suprious: %.2f' % (p_id, freq))
            del persons[p_id]
            continue

        # if (median_bbox_area) < SIZE_THR:
        #     print('Rejecting %d bc not big enough: %.2f' % (p_id,
        #                                                     median_bbox_area))
        #     del persons[p_id]
        #     continue
        if med_score < SCORE_THR:
            print('Rejecting %d bc not confident: %.2f' % (p_id, med_score))
            del persons[p_id]
            continue
        print('%d survived with: freq %.2f, score %.2f, size %.2f' % (p_id, freq, med_score, median_bbox_area))
    print('Total # of ppl trajectories: %d' % len(persons.keys()))
    if len(persons.keys()) == 0:
        return {}

    per_frame_smooth = smooth_detections(persons)
    per_frame = {}
    for p_id in persons.keys():
        # List of (t, bbox)
        # Make this into dict[t] = (p_id, bbox)
        for time, bbox, kp_here in persons[p_id]:
            if time in per_frame.keys():
                per_frame[time].append((p_id, bbox, kp_here))
            else:
                per_frame[time] = [(p_id, bbox, kp_here)]
    # Now show.
    if vis:#True:
        if not vis:
            frames = read_frames(vid_path)
        for i, frame in enumerate(frames):
            if i % 3 != 0:
                continue
            import matplotlib.pyplot as plt
            plt.ion()
            plt.clf()
            fig = plt.figure(1)
            ax = fig.add_subplot(111)
            frame = frames[i]
            ax.imshow(frame)
            ax.set_title('frame %d' % i)
            if i in per_frame.keys():
                people_here = per_frame[i]
                for p_id, bbox, kp_here in people_here:
                    skel_img = draw_openpose_skeleton(frame, kp_here)
                    ax.imshow(skel_img)
                    rect = get_rect(bbox, 'dashed')
                    ax.add_patch(rect)
                    plt.text(bbox[4], bbox[5], 'pid: %d' % p_id)
            if i in per_frame_smooth.keys():
                people_here = per_frame_smooth[i]
                for p_id, bbox, kp_here in people_here:
                    rect = get_rect(bbox, ecolor='blue')
                    ax.add_patch(rect)
                    plt.text(bbox[4], bbox[5], 'pid: %d' % p_id)
            plt.pause(1e-3)


    return per_frame_smooth


def smooth_detections(persons):
    # First fill in missing boxes.
    per_frame = {}
    for p_id in persons.keys():
        bboxes = persons[p_id]
        # for each person, get list of N x bbox
        start_fr = bboxes[0][0]
        end_fr = bboxes[-1][0]
        if len(bboxes) != (end_fr - start_fr):
            bboxeskp_filled = fill_in_bboxes(bboxes, start_fr, end_fr)
        else:
            bboxeskp_filled = [bbox[1:] for bbox in bboxes]
        # bboxes_filled is a list of tuple (bbox, kp) so split them
        bboxes_filled, kps_filled = [], []
        for bbox, kp in bboxeskp_filled:
            bboxes_filled.append(bbox)
            kps_filled.append(kp)

        # Now smooth this.
        times = np.arange(start_fr, end_fr)
        if len(bboxes_filled) == 0:
            continue

        bboxes_filled = np.vstack(bboxes_filled)
        kps_filled = np.stack(kps_filled)
        bbox_params = bboxes_filled[:, :3]
        bbox_scores = bboxes_filled[:, 3]
        # Filter the first 3 parameters (cx, cy, s)
        smoothed = np.array([signal.medfilt(param, 11) for param in bbox_params.T]).T
        from scipy.ndimage.filters import gaussian_filter1d
        smoothed2 = np.array([gaussian_filter1d(traj, 3) for traj in smoothed.T]).T

        # Convert the smoothed parameters into bboxes.
        smoothed_bboxes = np.vstack([params_to_bboxes(cx, cy, sc) for (cx, cy, sc) in smoothed2])
        # Cut back the boxes until confidence is high.
        last_ind = len(bbox_scores) - 1
        while bbox_scores[last_ind] < END_BOX_CONF:
            if last_ind <= 0:
                break
            last_ind -= 1
        # Make it into 8 dim (cx, cy, sc, score, x, y, h, w) again,,
        final_bboxes = np.hstack([smoothed2[:last_ind], bbox_scores.reshape(-1, 1)[:last_ind], smoothed_bboxes[:last_ind]])
        final_kps = kps_filled[:last_ind]

        # import matplotlib.pyplot as plt
        # plt.ion()
        # plt.figure(2)
        # plt.clf()
        # plt.subplot(311)
        # plt.plot(times, bbox_params[:, 0])
        # plt.plot(times, smoothed[:, 0])
        # plt.plot(times, smoothed2[:, 0])
        # plt.subplot(312)
        # plt.plot(times, bbox_params[:, 1])
        # plt.plot(times, smoothed[:, 1])
        # plt.plot(times, smoothed2[:,  1])
        # plt.subplot(313)
        # plt.plot(times, bbox_params[:, 2])
        # plt.plot(times, smoothed[:, 2])
        # plt.plot(times, smoothed2[:, 2])
        # plt.draw()
        # import ipdb; ipdb.set_trace()

        # Conver this into dict of time.
        for time, bbox, kps in zip(times, final_bboxes, final_kps):
            if time in per_frame.keys():
                per_frame[time].append((p_id, bbox, kps))
            else:
                per_frame[time] = [(p_id, bbox, kps)]

    return per_frame


def params_to_bboxes(cx, cy, scale):
    center = [cx, cy]
    radius = RADIUS * (1 / scale)
    top_corner = center - radius
    bbox = np.hstack([top_corner, radius * 2, radius * 2])

    return bbox


def fill_in_bboxes(bboxes, start_frame, end_frame):
    """
    bboxes is a list of (t, bbox, kps)
    remove gaps.
    """
    bboxes_filled = []
    bid = 0
    for i in range(start_frame, end_frame):
        if bboxes[bid][0] == i:
            bboxes_filled.append(bboxes[bid][1:])
            bid += 1
        else:
            # this time t doesnt exist!
            # Fill in with previous.
            fill_this = np.copy(bboxes_filled[-1])
            # but make sure that kp score is all 0
            fill_this[1][:, 2] = 0.
            bboxes_filled.append(fill_this)
                        

    return bboxes_filled
                
        
def get_rect(bbox0, linestyle='solid', ecolor='red'):
    """
    for drawing..
    bbox0 is (cx, cy, scale, score, x, y, h, w)
    """
    bbox = bbox0[-4:]
    return patches.Rectangle(
        (bbox[0], bbox[1]),
        bbox[2],
        bbox[3],
        linewidth=2,
        edgecolor=ecolor,
        linestyle=linestyle,
        fill=False,
        clip_on=False)


def compute_iou(bbox0, bboxes0):
    """
    bbox0 is (cx, cy, scale, score, x, y, h, w)
    last 4 bit is the standard bbox.
    For this ignore score.
    """

    def iou(boxA, boxB):
        boxA_area = boxA[2] * boxA[3]
        boxB_area = boxB[2] * boxB[3]
        min_x = max(boxA[0], boxB[0])
        min_y = max(boxA[1], boxB[1])
        endA = boxA[:2] + boxA[2:]
        endB = boxB[:2] + boxB[2:]
        max_x = min(endA[0], endB[0])
        max_y = max(endA[1], endB[1])
        w = max_x - min_x + 1
        h = max_y - min_y + 1
        inter_area = float(w * h)
        iou = max(0, inter_area / (boxA_area + boxB_area - inter_area))
        return iou

    return [iou(bbox0[-4:], bbox[-4:]) for bbox in bboxes0]


def read_json(json_path):
    with open(json_path) as f:
        data = json.load(f)
    kps = []
    for people in data['people']:
        kp = np.array(people['pose_keypoints']).reshape(-1, 3)
        kps.append(kp)
    return kps

def nonmaxsupp(bboxes0, valid_kps0):
    """
    bboxes are (cx, cy, scale, score, x, y, h, w)
    """
    if len(bboxes0) == 0:
        return [], []
    if bboxes0.shape[0] == 1:
        return bboxes0, valid_kps0
    pick = []
    scores = bboxes0[:, 3]
    bboxes = bboxes0[:, 4:]
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = x1 + bboxes[:, 2] - 1
    y2 = x2 + bboxes[:, 3] - 1
    area = bboxes[:, 2] * bboxes[:, 3]
    
    # Small first,,
    idxs = np.argsort(scores)

    while len(idxs) > 0:
        last = len(idxs)-1
        i = idxs[last]
        pick.append(i)
        # compute iou
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # Compute width height
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
	overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
	idxs = np.delete(idxs, np.concatenate(([last],
			                       np.where(overlap > NMS_THR)[0])))

    return bboxes0[pick], valid_kps0[pick]
                
def get_bbox(kp):
    vis = kp[:, 2] > VIS_THR
    if np.sum(vis) < NUM_VIS_THR:
        return None, None
    vis_kp = kp[vis, :2]
    min_pt = np.min(vis_kp, axis=0)
    max_pt = np.max(vis_kp, axis=0)
    person_height = np.linalg.norm(max_pt - min_pt)
    if person_height == 0:
        print('bad!')
        import ipdb; ipdb.set_trace()
    center = (min_pt + max_pt) / 2.
    scale = 150. / person_height

    score = np.sum(kp[vis, 2]) / np.sum(vis)

    radius = RADIUS * (1 / scale)
    top_corner = center - radius
    bbox = np.hstack([top_corner, radius * 2, radius * 2])

    return np.hstack([center, scale, score, bbox]), kp


def read_frames(path, max_num=None):
    vid = cv2.VideoCapture(path)

    imgs = []
    # n_frames = int(vid.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    counter = 0
    success = True
    while success:
        success, img = vid.read()
        if success:
            # Make BGR->RGB!!!!
            imgs.append(img[:, :, ::-1])
            counter += 1
            if max_num is not None and counter >= max_num:
                break

    vid.release()

    return imgs


if __name__ == '__main__':
    tf.app.run()
