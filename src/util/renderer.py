"""
Renders mesh using OpenDr for visualization.
+ 2D drawing functions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2

from opendr.camera import ProjectPoints
from opendr.renderer import ColoredRenderer
from opendr.lighting import LambertianPointLight

colors = {
    # colorblind/print/copy safe:
    'light_blue': [0.65098039, 0.74117647, 0.85882353],
    'light_pink': [.9, .7, .7],  # This is used to do no-3d
}


class SMPLRenderer(object):
    def __init__(self,
                 img_size=224,
                 flength=500.,
                 face_path="tf_smpl/smpl_faces.npy"):
        self.faces = np.load(face_path)
        self.w = img_size
        self.h = img_size
        self.flength = flength

    def __call__(self,
                 verts,
                 cam=None,
                 img=None,
                 do_alpha=False,
                 far=None,
                 near=None,
                 color_id=0,
                 img_size=None):
        """
        cam is 3D [f, px, py]
        """
        if img is not None:
            h, w = img.shape[:2]
        elif img_size is not None:
            h = img_size[0]
            w = img_size[1]
        else:
            h = self.h
            w = self.w

        if cam is None:
            cam = [self.flength, w / 2., h / 2.]

        use_cam = ProjectPoints(
            f=cam[0] * np.ones(2),
            rt=np.zeros(3),
            t=np.zeros(3),
            k=np.zeros(5),
            c=cam[1:3])

        if near is None:
            near = np.maximum(np.min(verts[:, 2]) - 25, 0.1)
        if far is None:
            far = np.maximum(np.max(verts[:, 2]) + 25, 25)

        imtmp = render_model(
            verts,
            self.faces,
            w,
            h,
            use_cam,
            do_alpha=do_alpha,
            img=img,
            far=far,
            near=near,
            color_id=color_id)

        return (imtmp * 255).astype('uint8')

    def rotated(self,
                verts,
                deg,
                cam=None,
                axis='y',
                img=None,
                do_alpha=True,
                far=None,
                near=None,
                color_id=0,
                img_size=None):
        import math
        if axis == 'y':
            around = cv2.Rodrigues(np.array([0, math.radians(deg), 0]))[0]
        elif axis == 'x':
            around = cv2.Rodrigues(np.array([math.radians(deg), 0, 0]))[0]
        else:
            around = cv2.Rodrigues(np.array([0, 0, math.radians(deg)]))[0]
        center = verts.mean(axis=0)
        new_v = np.dot((verts - center), around) + center

        return self.__call__(
            new_v,
            cam,
            img=img,
            do_alpha=do_alpha,
            far=far,
            near=near,
            img_size=img_size,
            color_id=color_id)


def _create_renderer(w=640,
                     h=480,
                     rt=np.zeros(3),
                     t=np.zeros(3),
                     f=None,
                     c=None,
                     k=None,
                     near=.5,
                     far=10.):

    f = np.array([w, w]) / 2. if f is None else f
    c = np.array([w, h]) / 2. if c is None else c
    k = np.zeros(5) if k is None else k

    rn = ColoredRenderer()

    rn.camera = ProjectPoints(rt=rt, t=t, f=f, c=c, k=k)
    rn.frustum = {'near': near, 'far': far, 'height': h, 'width': w}
    return rn


def _rotateY(points, angle):
    """Rotate the points by a specified angle."""
    ry = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
                   [-np.sin(angle), 0., np.cos(angle)]])
    return np.dot(points, ry)


def simple_renderer(rn,
                    verts,
                    faces,
                    yrot=np.radians(120),
                    color=colors['light_pink']):
    # Rendered model color
    rn.set(v=verts, f=faces, vc=color, bgcolor=np.ones(3))
    albedo = rn.vc

    # Construct Back Light (on back right corner)
    rn.vc = LambertianPointLight(
        f=rn.f,
        v=rn.v,
        num_verts=len(rn.v),
        light_pos=_rotateY(np.array([-200, -100, -100]), yrot),
        vc=albedo,
        light_color=np.array([1, 1, 1]))

    # Construct Left Light
    rn.vc += LambertianPointLight(
        f=rn.f,
        v=rn.v,
        num_verts=len(rn.v),
        light_pos=_rotateY(np.array([800, 10, 300]), yrot),
        vc=albedo,
        light_color=np.array([1, 1, 1]))

    # Construct Right Light
    rn.vc += LambertianPointLight(
        f=rn.f,
        v=rn.v,
        num_verts=len(rn.v),
        light_pos=_rotateY(np.array([-500, 500, 1000]), yrot),
        vc=albedo,
        light_color=np.array([.7, .7, .7]))

    return rn.r


def get_alpha(imtmp, bgval=1.):
    h, w = imtmp.shape[:2]
    alpha = (~np.all(imtmp == bgval, axis=2)).astype(imtmp.dtype)

    b_channel, g_channel, r_channel = cv2.split(imtmp)

    im_RGBA = cv2.merge((b_channel, g_channel, r_channel, alpha.astype(
        imtmp.dtype)))
    return im_RGBA


def append_alpha(imtmp):
    alpha = np.ones_like(imtmp[:, :, 0]).astype(imtmp.dtype)
    if np.issubdtype(imtmp.dtype, np.uint8):
        alpha = alpha * 255
    b_channel, g_channel, r_channel = cv2.split(imtmp)
    im_RGBA = cv2.merge((b_channel, g_channel, r_channel, alpha))
    return im_RGBA


def render_model(verts,
                 faces,
                 w,
                 h,
                 cam,
                 near=0.5,
                 far=25,
                 img=None,
                 do_alpha=False,
                 color_id=None):
    rn = _create_renderer(
        w=w, h=h, near=near, far=far, rt=cam.rt, t=cam.t, f=cam.f, c=cam.c)

    # Uses img as background, otherwise white background.
    if img is not None:
        rn.background_image = img / 255. if img.max() > 1 else img

    if color_id is None:
        color = colors['light_blue']
    else:
        color_list = colors.values()
        color = color_list[color_id % len(color_list)]

    imtmp = simple_renderer(rn, verts, faces, color=color)

    # If white bg, make transparent.
    if img is None and do_alpha:
        imtmp = get_alpha(imtmp)
    elif img is not None and do_alpha:
        imtmp = append_alpha(imtmp)

    return imtmp


# ------------------------------


def get_original(proc_param, verts, cam, joints):
    # img_size is the size it was cropped to.
    img_size = proc_param['target_size']
    undo_scale = 1. / np.array(proc_param['scale'])

    cam_s = cam[0]
    cam_pos = cam[1:]
    principal_pt = np.array([img_size, img_size]) / 2.
    flength = 500.
    tz = flength / (0.5 * img_size * cam_s)
    trans = np.hstack([cam_pos, tz])
    vert_shifted = verts + trans

    start_pt = proc_param['start_pt']# - 0.5 * img_size
    final_principal_pt = (principal_pt + start_pt) * undo_scale
    cam_for_render = np.hstack(
        [np.mean(flength * undo_scale), final_principal_pt])

    joints = ((joints + 1) * 0.5) * img_size    
    # This is in padded image.
    kp_original = (joints + proc_param['start_pt']) * undo_scale
    # Subtract padding from joints.
    # margin = int(img_size / 2)
    # kp_original = (joints + proc_param['start_pt'] - margin) * undo_scale

    return cam_for_render, vert_shifted, kp_original


def render_original(frame,
                    skel_frame,
                    proc_param,
                    result,
                    other_vp,
                    other_vp2,
                    bbox,
                    renderer,
                    ppl_id=0):
    """
    Render the result in the original coordinate frame
    ppl_id is optional for color.
    """
    verts = result['verts']
    joints = result['joints']
    cam = result['cams']
    img_size = frame.shape[:2]
    cam_for_render, vert_shifted, kp_original = get_original(
        proc_param, verts, cam, joints)

    # Draw 2D projection.
    radius = (np.mean(frame.shape[:2]) * 0.01).astype(int)
    draw_bbox(skel_frame, bbox[-4:])
    skel_img_orig = draw_skeleton(skel_frame, kp_original, radius=radius)
    # Rendered image.
    rend_img_orig = renderer(
        vert_shifted, cam=cam_for_render, img=frame, color_id=ppl_id)
    # Another viewpoint!
    another_vp = renderer.rotated(
        vert_shifted,
        60,
        cam=cam_for_render,
        color_id=ppl_id,
        img=other_vp,
        axis='y')
    another_vp2 = renderer.rotated(
        vert_shifted,
        -60,
        cam=cam_for_render,
        color_id=ppl_id,
        img=other_vp2,
        axis='x')

    # import matplotlib.pyplot as plt
    # plt.ion()
    # plt.clf()
    # plt.imshow(skel_img_orig)
    # plt.imshow(rend_img_orig)
    # plt.draw()
    # row1 = np.hstack((skel_img_orig, rend_img_orig))
    # row2 = np.hstack((another_vp, another_vp2))[:, :, :3]
    # plt.imshow(np.vstack((row1, row2)).astype(np.uint8))

    return rend_img_orig, skel_img_orig, another_vp, another_vp2


# --------------------
# 2D draw functions
# --------------------

def draw_skeleton(input_image, joints, draw_edges=True, vis=None, radius=None):
    """
    joints is 3 x 19. but if not will transpose it.
    0: Right ankle
    1: Right knee
    2: Right hip
    3: Left hip
    4: Left knee
    5: Left ankle
    6: Right wrist
    7: Right elbow
    8: Right shoulder
    9: Left shoulder
    10: Left elbow
    11: Left wrist
    12: Neck
    13: Head top
    14: nose
    15: left_eye
    16: right_eye
    17: left_ear
    18: right_ear
    """
    import numpy as np
    import cv2

    if radius is None:
        radius = max(4, (np.mean(input_image.shape[:2]) * 0.01).astype(int))

    colors = {
        'pink': np.array([197, 27, 125]),  # L lower leg
        'light_pink': np.array([233, 163, 201]),  # L upper leg
        'light_green': np.array([161, 215, 106]),  # L lower arm
        'green': np.array([77, 146, 33]),  # L upper arm
        'red': np.array([215, 48, 39]),  # head
        'light_red': np.array([252, 146, 114]),  # head
        'light_orange': np.array([252, 141, 89]),  # chest
        'purple': np.array([118, 42, 131]),  # R lower leg
        'light_purple': np.array([175, 141, 195]),  # R upper
        'light_blue': np.array([145, 191, 219]),  # R lower arm
        'blue': np.array([69, 117, 180]),  # R upper arm
        'gray': np.array([130, 130, 130]),  #
        'white': np.array([255, 255, 255]),  #
    }

    image = input_image.copy()
    input_is_float = False

    if np.issubdtype(image.dtype, np.float):
        input_is_float = True
        max_val = image.max()
        if max_val <= 2.:  # should be 1 but sometimes it's slightly above 1
            image = (image * 255).astype(np.uint8)
        else:
            image = (image).astype(np.uint8)

    if joints.shape[0] != 2:
        joints = joints.T
    joints = np.round(joints).astype(int)

    jcolors = [
        'light_pink', 'light_pink', 'light_pink', 'pink', 'pink', 'pink',
        'light_blue', 'light_blue', 'light_blue', 'blue', 'blue', 'blue',
        'purple', 'purple', 'red', 'green', 'green', 'white', 'white'
    ]

    if joints.shape[1] == 19:
        # parent indices -1 means no parents
        parents = np.array([
            1, 2, 8, 9, 3, 4, 7, 8, 12, 12, 9, 10, 14, -1, 13, -1, -1, 15, 16
        ])
        # Left is light and right is dark
        ecolors = {
            0: 'light_pink',
            1: 'light_pink',
            2: 'light_pink',
            3: 'pink',
            4: 'pink',
            5: 'pink',
            6: 'light_blue',
            7: 'light_blue',
            8: 'light_blue',
            9: 'blue',
            10: 'blue',
            11: 'blue',
            12: 'purple',
            17: 'light_green',
            18: 'light_green',
            14: 'purple'
        }
    elif joints.shape[1] == 19:
        parents = np.array([
            1,
            2,
            8,
            9,
            3,
            4,
            7,
            8,
            -1,
            -1,
            9,
            10,
            13,
            -1,
        ])
        ecolors = {
            0: 'light_pink',
            1: 'light_pink',
            2: 'light_pink',
            3: 'pink',
            4: 'pink',
            5: 'pink',
            6: 'light_blue',
            7: 'light_blue',
            10: 'light_blue',
            11: 'blue',
            12: 'purple'
        }
    else:
        print('Unknown skeleton!!')
        import ipdb
        ipdb.set_trace()

    for child in xrange(len(parents)):
        point = joints[:, child]
        # If invisible skip
        if vis is not None and vis[child] == 0:
            continue
        if draw_edges:
            cv2.circle(image, (point[0], point[1]), radius, colors['white'],
                       -1)
            cv2.circle(image, (point[0], point[1]), radius - 1,
                       colors[jcolors[child]], -1)
        else:
            # cv2.circle(image, (point[0], point[1]), 5, colors['white'], 1)
            cv2.circle(image, (point[0], point[1]), radius - 1,
                       colors[jcolors[child]], 1)
            # cv2.circle(image, (point[0], point[1]), 5, colors['gray'], -1)
        pa_id = parents[child]
        if draw_edges and pa_id >= 0:
            if vis is not None and vis[pa_id] == 0:
                continue
            point_pa = joints[:, pa_id]
            cv2.circle(image, (point_pa[0], point_pa[1]), radius - 1,
                       colors[jcolors[pa_id]], -1)
            if child not in ecolors.keys():
                print('bad')
                import ipdb
                ipdb.set_trace()
            cv2.line(image, (point[0], point[1]), (point_pa[0], point_pa[1]),
                     colors[ecolors[child]], radius - 2)

    # Convert back in original dtype
    if input_is_float:
        if max_val <= 1.:
            image = image.astype(np.float32) / 255.
        else:
            image = image.astype(np.float32)

    return image


def draw_openpose_skeleton(input_image, joints, draw_edges=True, vis=None, radius=None, threshold=0.1):
    """
    Openpose:
    joints is 3 x 18. but if not will transpose it.
    0: Nose
    1: Neck
    2: R Should
    3: R Elbow
    4: R Wrist
    5: L Should
    6: L Elbow
    7: L Wrist
    8: R Hip
    9: R Knee
    10: R Ankle
    11: L Hip
    12: L Knee
    13: L Ankle
    14: R Eye
    15: L Eye
    16: R Ear
    17: L Ear

    [18: Head (always 0)]
    This function converts the order into my lsp-coco skeleton (19 points) and uses that draw function.    
    """
    if joints.shape[0] != 2:
        joints = joints.T
    # Append a dummy 0 joint for the head.
    joints = np.hstack((joints, np.zeros((3,1))))
    # # Figure out the order:
    # for i in range(joints.shape[1]):
    #     import matplotlib.pyplot as plt
    #     plt.ion()
    #     plt.clf()
    #     fig = plt.figure(1)
    #     plt.imshow(input_image)
    #     plt.scatter(joints[0, :], joints[1, :])
    #     plt.scatter(joints[0, i], joints[1, i])
    #     plt.title('%dth' % i)
    #     import ipdb; ipdb.set_trace()

    # This is what we want.
    joint_names = ['R Ankle',
                   'R Knee',
                   'R Hip',
                   'L Hip',
                   'L Knee',
                   'L Ankle',
                   'R Wrist',
                   'R Elbow',
                   'R Shoulder',
                   'L Shoulder',
                   'L Elbow',
                   'L Wrist',
                   'Neck',
                   'Head',
                   'Nose', 'L Eye', 'R Eye', 'L Ear', 'R Ear']
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

    ordered_joints = joints[:, permute_order]

    vis = ordered_joints[2, :] > threshold

    # ordered_joints = np.vstack((ordered_joints[:2, :], vis))

    image = draw_skeleton(input_image, ordered_joints[:2, :], draw_edges, vis, radius)
    # import matplotlib.pyplot as plt
    # plt.ion()
    # plt.clf()
    # plt.imshow(image)
    # plt.draw()

    return image


def draw_bbox(frame, bbox):
    # bbox is [x, y, h, w]
    x, y = (bbox[:2]).astype(np.int)
    width, height = (bbox[2:]).astype(np.int)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (158, 202, 225), 2)
