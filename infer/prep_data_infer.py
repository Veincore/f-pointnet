import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # '../frustum-pointnets/infer'
ROOT_DIR = os.path.dirname(BASE_DIR)  # '../frustum-pointnets'
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'mayavi'))
#sys.path.append(os.path.join(ROOT_DIR, 'train'))

if os.path.exists('/opt/ros/kinetic/lib/python2.7/dist-packages/'):
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import argparse
import time
import cv2
import numpy as np
import mxnet as mx
import importlib
import matplotlib.pyplot as plt
import gluoncv
import kitti.kitti_util as utils
from train.test import test_from_rgb_detection, get_session_and_ops

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
parser.add_argument('--model', default='frustum_pointnets_v1', help='Model name [default: frustum_pointnets_v1]')
parser.add_argument('--model_path', default='log/model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--batch_size', type=int, default=32, help='batch size for inference [default: 32]')
parser.add_argument('--output', default='test_results', help='output file/folder name [default: test_results]')
parser.add_argument('--data_path', default=None, help='frustum dataset pickle filepath [default: None]')
parser.add_argument('--from_rgb_detection', action='store_true', help='test from dataset files from rgb detection.')
parser.add_argument('--idx_path', default=None, help='filename of txt where each line is a data idx, used for rgb detection -- write <id>.txt for all frames. [default: None]')
parser.add_argument('--dump_result', action='store_true', help='If true, also dump results to .pickle file')
FLAGS = parser.parse_args()

# Set training configurations
BATCH_SIZE = FLAGS.batch_size
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
NUM_POINT = FLAGS.num_point  # 1024
MODEL = importlib.import_module(FLAGS.model)
NUM_CLASSES = 2
NUM_CHANNEL = 4

#raw_input = input()

class calib_infer():
    ''' Calibration matrices and utils
            3d XYZ in <label>.txt are in rect camera coord.
            2d box xy are in image2 coord
            Points in <lidar>.bin are in Velodyne coord.

            y_image2 = P^2_rect * x_rect
            y_image2 = P^2_rect * R0_rect * Tr_velo_to_cam * x_velo
            x_ref = Tr_velo_to_cam * x_velo
            x_rect = R0_rect * x_ref

            P^2_rect = [f^2_u,  0,      c^2_u,  -f^2_u b^2_x;
                        0,      f^2_v,  c^2_v,  -f^2_v b^2_y;
                        0,      0,      1,      0]
                     = K * [1|t]

            image2 coord:
             ----> x-axis (u)
            |
            |
            v y-axis (v)

            velodyne coord:
            front x, left y, up z

            rect/ref camera coord:
            right x, down y, front z
        '''
    def __init__(self, calib_dir):
        calibs = self.read_calib_file(calib_dir)
        # Tr_velo_to_cam [4, 4]
        self.V2C = np.zeros([3, 4])
        self.V2C[:, :3] = np.reshape(calibs['R'], [3, 3])
        self.V2C[:, 3:4] = np.reshape(calibs['T'], [3, 1])
        self.C2V = utils.inverse_rigid_trans(self.V2C)
        # P2
        self.P = np.reshape(calibs['P_rect_02'], [3,4])
        # R0
        self.R0 = np.reshape(calibs['R_rect_00'], [3,3])
        # Camera intrinsics and extrinsics
        self.c_u = self.P[0, 2]
        self.c_v = self.P[1, 2]
        self.f_u = self.P[0, 0]
        self.f_v = self.P[1, 1]
        self.b_x = self.P[0, 3] / (-self.f_u)  # relative
        self.b_y = self.P[1, 3] / (-self.f_v)

    def read_calib_file(self, calib_dir):
        data = {}
        cam_to_cam_file = os.path.join(calib_dir, 'calib_cam_to_cam.txt')
        velo_to_cam_file = os.path.join(calib_dir, 'calib_velo_to_cam.txt')
        with open(cam_to_cam_file, 'r') as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line)==0: continue
                key, value = line.split(':', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
        with open(velo_to_cam_file, 'r') as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line)==0: continue
                key, value = line.split(':', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
        return data

    def cart2hom(self, pts_3d):
        ''' Input: nx3 points in Cartesian
            Oupput: nx4 points in Homogeneous by pending 1
        '''
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n,1))))
        return pts_3d_hom

    # ===========================
    # ------- 3d to 3d ----------
    # ===========================
    def project_velo_to_ref(self, pts_3d_velo):
        pts_3d_velo = self.cart2hom(pts_3d_velo)  # nx4
        return np.dot(pts_3d_velo, np.transpose(self.V2C))

    def project_ref_to_velo(self, pts_3d_ref):
        pts_3d_ref = self.cart2hom(pts_3d_ref)  # nx4
        return np.dot(pts_3d_ref, np.transpose(self.C2V))

    def project_rect_to_ref(self, pts_3d_rect):
        ''' Input and Output are nx3 points '''
        return np.transpose(np.dot(np.linalg.inv(self.R0), np.transpose(pts_3d_rect)))

    def project_ref_to_rect(self, pts_3d_ref):
        ''' Input and Output are nx3 points '''
        return np.transpose(np.dot(self.R0, np.transpose(pts_3d_ref)))

    def project_rect_to_velo(self, pts_3d_rect):
        ''' Input: nx3 points in rect camera coord.
            Output: nx3 points in velodyne coord.
        '''
        pts_3d_ref = self.project_rect_to_ref(pts_3d_rect)
        return self.project_ref_to_velo(pts_3d_ref)

    def project_velo_to_rect(self, pts_3d_velo):
        pts_3d_ref = self.project_velo_to_ref(pts_3d_velo)
        return self.project_ref_to_rect(pts_3d_ref)

    # ===========================
    # ------- 3d to 2d ----------
    # ===========================
    def project_rect_to_image(self, pts_3d_rect):
        ''' Input: nx3 points in rect camera coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_rect = self.cart2hom(pts_3d_rect)
        pts_2d = np.dot(pts_3d_rect, np.transpose(self.P))  # nx3
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        return pts_2d[:, 0:2]

    def project_velo_to_image(self, pts_3d_velo):
        ''' Input: nx3 points in velodyne coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_rect = self.project_velo_to_rect(pts_3d_velo)
        return self.project_rect_to_image(pts_3d_rect)

    # ===========================
    # ------- 2d to 3d ----------
    # ===========================
    def project_image_to_rect(self, uv_depth):
        ''' Input: nx3 first two channels are uv, 3rd channel
                   is depth in rect camera coord.
            Output: nx3 points in rect camera coord.
        '''
        n = uv_depth.shape[0]
        x = ((uv_depth[:, 0] - self.c_u) * uv_depth[:, 2]) / self.f_u + self.b_x
        y = ((uv_depth[:, 1] - self.c_v) * uv_depth[:, 2]) / self.f_v + self.b_y
        pts_3d_rect = np.zeros((n, 3))
        pts_3d_rect[:, 0] = x
        pts_3d_rect[:, 1] = y
        pts_3d_rect[:, 2] = uv_depth[:, 2]
        return pts_3d_rect

    def project_image_to_velo(self, uv_depth):
        pts_3d_rect = self.project_image_to_rect(uv_depth)
        return self.project_rect_to_velo(pts_3d_rect)


class kitti_object_infer():
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.num_samples = 109

        self.image_dir = os.path.join(self.root_dir, 'image_02/data')
        self.lidar_dir = os.path.join(self.root_dir, 'velodyne_points/data')
        self.calib_dir = os.path.join(self.root_dir, '2011_09_26_calib/2011_09_26')
        # self.image_dir = os.path.join(self.root_dir, 'image_02\\data')
        # self.calib_dir = os.path.join(self.root_dir, '2011_09_26_calib\\2011_09_26')
        # self.lidar_dir = os.path.join(self.root_dir, 'velodyne_points\\data')

    def __len__(self):
        return self.num_samples

    def get_image(self, idx):
        assert(idx < self.num_samples)
        img_filename = os.path.join(self.image_dir, '%010d.png'%(idx))
        #img_filename = os.path.join(self.image_dir, '0000000000.png')
        print('filename: ', img_filename)
        return utils.load_image(img_filename), img_filename

    def get_lidar(self, idx):
        assert(idx < self.num_samples)
        lidar_filename = os.path.join(self.lidar_dir, '%010d.bin'%(idx))
        #lidar_filename = os.path.join(self.lidar_dir, '0000000000.bin')
        return utils.load_velo_scan(lidar_filename)

    def get_calibration(self):
        return calib_infer(self.calib_dir)

def get_lidar_in_image_fov(pc_velo, calib, xmin, ymin, xmax, ymax,
                           return_more=False, clip_distance=2.0):
    ''' Filter lidar points, keep those in image FOV '''
    pts_2d = calib.project_velo_to_image(pc_velo)
    fov_inds = (pts_2d[:, 0] < xmax) & (pts_2d[:, 0] >= xmin) & \
        (pts_2d[:, 1] < ymax) & (pts_2d[:, 1] >= ymin)
    fov_inds = fov_inds & (pc_velo[:, 0] > clip_distance)
    imgfov_pc_velo = pc_velo[fov_inds, :]
    if return_more:
        return imgfov_pc_velo, pts_2d, fov_inds
    else:
        return imgfov_pc_velo  # [m, 4]

def show_lidar(pc_velo, calib, fig, img_fov=False, img_width=None, img_height=None):
    ''' Show all LiDAR points.
        Draw 3d box in LiDAR point cloud (in velo coord system) '''
    if 'mlab' not in sys.modules: import mayavi.mlab as mlab
    from viz_util import draw_lidar

    #mlab.clf(fig)
    print(('All point num: ', pc_velo.shape[0]))
    #fig = mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(1000, 500))
    if img_fov:
        pc_velo = get_lidar_in_image_fov(pc_velo, calib, 0, 0,
            img_width, img_height)
        print(('FOV point num: ', pc_velo.shape[0]))
    draw_lidar(pc_velo, fig=fig)
    mlab.show(30)

def show_lidar_on_image(pc_velo, img, calib, img_width, img_height):
    ''' Project LiDAR points to image '''
    imgfov_pc_velo, pts_2d, fov_inds = get_lidar_in_image_fov(pc_velo,
        calib, 0, 0, img_width, img_height, True)
    imgfov_pts_2d = pts_2d[fov_inds,:]
    imgfov_pc_rect = calib.project_velo_to_rect(imgfov_pc_velo)

    cmap = plt.cm.get_cmap('hsv', 256)
    cmap = np.array([cmap(i) for i in range(256)])[:,:3]*255

    for i in range(imgfov_pts_2d.shape[0]):
        depth = imgfov_pc_rect[i,2]
        color = cmap[int(640.0/depth),:]
        cv2.circle(img, (int(np.round(imgfov_pts_2d[i,0])),
            int(np.round(imgfov_pts_2d[i,1]))),
            2, color=tuple(color), thickness=-1)
    #Image.fromarray(img).show()
    cv2.imshow('lidar on image', img)
    cv2.waitKey(30)
    return img

def transform_bbox_inverse(bbox_lists, img_ori_shape, img_shape):
    # 将yolo得出的bbox映射回原图像
    # img_shape: (w, h, c) , inputs of YOLO
    # img_ori_shape: (w, h, c), origin image shape
    w_ori, h_ori, _ = img_ori_shape
    w, h, _ = img_shape
    scale_w = w_ori / w
    scale_h = h_ori / h
    bbox_lists[:, 0] *= scale_w
    bbox_lists[:, 2] *= scale_w
    bbox_lists[:, 1] *= scale_h
    bbox_lists[:, 3] *= scale_h

    bbox_lists = bbox_lists.astype(int)
    return bbox_lists

def get_2d_box_yolo(img, net):
    '''
    :param img: ndarray, BGR
            net: gluoncv model_zoo

    :return:  NDArray (mxnet)
            class_IDs: [batch, 100, 1], 使用时仅用{'Car_6': 0, 'Pedestrian_14': 1, 'Cyclist_1': 2}
                   0        1      2    3     4     5   6   7   8     9      10
                aeroplane bicycle bird boat bottle bus car cat chair cow diningtable
                11   12      13       14       15       16    17   18      19
                dog horse motorbike person pottedplant sheep sofa train tvmonitor

            scores: [batch, 100, 1]
            bounding_boxes: [batch, 100, 4], [xmin, ymin, xmax, ymax]
    '''
    #net = gluoncv.model_zoo.get_model('yolo3_darknet53_voc', pretrained=True)
    img_ori = img
    img = mx.nd.array(img[:, :, ::-1])
    x, img = gluoncv.data.transforms.presets.yolo.transform_test(img, short = 512)
    class_IDs, scores, bounding_boxs = net(x.as_in_context(mx.gpu(0)))

    # 选出检测到的物体
    class_IDs, scores, bounding_boxs = class_IDs.asnumpy(), scores.asnumpy(), bounding_boxs.asnumpy()
    class_id_index = np.where(class_IDs > -1)
    class_IDs = class_IDs[class_id_index]
    scores = scores[class_id_index]
    bounding_boxs = bounding_boxs[:, :len(class_IDs), :].squeeze(0)

    # 去掉车、人、自行车以外的物体
    class_id_index = [i for i, e in enumerate(class_IDs) if e in [6, 14, 1]]
    class_IDs = class_IDs[class_id_index]
    scores = scores[class_id_index]
    bounding_boxs = bounding_boxs[class_id_index, :]
    bounding_boxs = transform_bbox_inverse(bounding_boxs, img_ori.shape, img.shape)


    return class_IDs, scores, bounding_boxs

def extract_data(dataset, net, data_idx):
    type_whitelist = [6, 14, 1] # 6:car 14: person 1:bicycle
    id_list = []
    box2d_list = [] # [xmin,ymin,xmax,ymax]
    type_list = []
    prob_list = []
    input_list = []  # channel number = 4, xyz,intensity in rect camera coord
    frustum_angle_list = []

    img, _ = dataset.get_image(data_idx)
    calib = dataset.get_calibration()
    pc_velo = dataset.get_lidar(data_idx)
    pc_rect = np.zeros_like(pc_velo)
    pc_rect[:, 0:3] = calib.project_velo_to_rect(pc_velo[:, 0:3])

    pc_rect[:, 3] = pc_velo[:, 3]
    img_height, img_width, img_channel = img.shape
    det_type_list, det_prob_list, det_box2d_list = get_2d_box_yolo(img, net)  # 0.8s
    show_image_with_2d_boxes(img, det_box2d_list)

    _, pc_image_coord, img_fov_inds = get_lidar_in_image_fov( \
        pc_velo[:, 0:3], calib, 0, 0, img_width, img_height, True)

    for obj_idx in range(len(det_type_list)):
        if det_type_list[obj_idx] not in type_whitelist : continue

        box2d = det_box2d_list[obj_idx]
        xmin, ymin, xmax, ymax = box2d
        box_fov_inds = (pc_image_coord[:, 0] < xmax) & \
                       (pc_image_coord[:, 0] >= xmin) & \
                       (pc_image_coord[:, 1] < ymax) & \
                       (pc_image_coord[:, 1] >= ymin)
        box_fov_inds = box_fov_inds & img_fov_inds
        pc_in_box_fov = pc_rect[box_fov_inds, :]
        #print('pc_in_fov: ', pc_in_box_fov.shape[0])
        if pc_in_box_fov.shape[0] == 0:
            continue
        # get frustum angle
        box2d_center = np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0])
        uvdepth = np.zeros((1, 3))
        uvdepth[0, 0:2] = box2d_center
        uvdepth[0, 2] = 20  # some random depth
        box2d_center_rect = calib.project_image_to_rect(uvdepth)
        frustum_angle = -1 * np.arctan2(box2d_center_rect[0, 2],
                                        box2d_center_rect[0, 0])

        id_list.append(obj_idx)
        box2d_list.append(np.array([xmin,ymin,xmax,ymax]))
        input_list.append(pc_in_box_fov)
        type_list.append(det_type_list[obj_idx])
        prob_list.append(det_prob_list[obj_idx])
        frustum_angle_list.append(frustum_angle)

    data = {}
    data['id_list'] = id_list
    data['box2d'] = box2d_list
    data['pc_in_box'] = input_list
    data['type'] = type_list
    data['prob'] = prob_list
    data['frustum_angle'] = frustum_angle_list
    return data

def rotate_pc_along_y(pc, rot_angle):
    '''
    Input:
        pc: numpy array (N,C), first 3 channels are XYZ
            z is facing forward, x is left ward, y is downward
        rot_angle: rad scalar
    Output:
        pc: updated pc with XYZ rotated
    '''
    cosval = np.cos(rot_angle)
    sinval = np.sin(rot_angle)
    rotmat = np.array([[cosval, -sinval],[sinval, cosval]])
    pc[:,[0,2]] = np.dot(pc[:,[0,2]], np.transpose(rotmat))
    return pc

class frustum_data_infer():
    def __init__(self, data, npoints, random_flip = False, random_shift = False,
                 rotate_to_center = False, one_hot = False):
        self.npoints = npoints
        self.random_flip = random_flip
        self.random_shift = random_shift
        self.rotate_to_center = rotate_to_center
        self.one_hot = one_hot

        self.id_list = data['id_list']
        self.box2d_list = data['box2d']
        self.input_list = data['pc_in_box']
        self.type_list = data['type']
        self.frustum_angle_list = data['frustum_angle']
        self.prob_list = data['prob']

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, index):
        rot_angle = self.get_center_view_rot_angle(index)
        # Compute one hot vector
        type2onehotclass = {'6': 0, '14': 1, '1': 2}
        if self.one_hot:
            cls_type = str(int(self.type_list[index]))
            # print('cls_type: ', cls_type)
            assert (cls_type in ['6', '14', '1'])
            one_hot_vec = np.zeros((3))
            one_hot_vec[type2onehotclass[cls_type]] = 1

        # Get point cloud
        if self.rotate_to_center:
            point_set = self.get_center_view_point_set(index)
        else:
            point_set = self.input_list[index]

            # Resample
        if point_set.shape[0] > 0:
            choice = np.random.choice(point_set.shape[0], self.npoints, replace=True)
            point_set = point_set[choice, :]

        if self.one_hot:
            return point_set, rot_angle, self.prob_list[index], one_hot_vec
        else:
            return point_set, rot_angle, self.prob_list[index]

    def get_center_view_rot_angle(self, index):
        ''' Get the frustum rotation angle, it isshifted by pi/2 so that it
        can be directly used to adjust GT heading angle '''
        return np.pi/2.0 + self.frustum_angle_list[index]

    def get_center_view_point_set(self, index):
        ''' Frustum rotation of point clouds.
        NxC points with first 3 channels as XYZ
        z is facing forward, x is left ward, y is downward
        '''
        # Use np.copy to avoid corrupting original data
        point_set = np.copy(self.input_list[index])
        return rotate_pc_along_y(point_set, \
            self.get_center_view_rot_angle(index))

def show_image_with_2d_boxes(img, box_list):
    for box in box_list:
        cv2.rectangle(img, (int(box[0]),int(box[1])),
            (int(box[2]),int(box[3])), (0,255,0), 2)
    cv2.imshow('img_with_box', img)
    cv2.waitKey(30)

def demo():
    if 'mlab' not in sys.modules: import mayavi.mlab as mlab
    from viz_util import draw_gt_boxes3d

    dataset = kitti_object_infer('/media/vdc/backup/database_backup/Chris/f-pointnet/2011_09_26_drive_0001_sync')
    calibs = dataset.get_calibration()
    #calibs = calib_infer('/media/vdc/backup/database_backup/Chris/f-pointnet/2011_09_26_drive_0001_sync/2011_09_26_calib/2011_09_26')
    #dataset = kitti_object_infer('D:\\Detectron_Data\\2011_09_26_drive_0001_sync')
    net = gluoncv.model_zoo.get_model('yolo3_darknet53_voc', pretrained=True, ctx=mx.gpu(0))
    sess, ops = get_session_and_ops(batch_size=BATCH_SIZE, num_point=NUM_POINT)
    fig = mlab.figure(figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1000, 500))
    for i in range(len(dataset)) :
        time1 = time.time()
        data = extract_data(dataset, net, i)  # 0.9s
        TEST_DATASET = frustum_data_infer(data, 1024, rotate_to_center=True, one_hot=True) # us级
        box_3d_list = test_from_rgb_detection(TEST_DATASET, sess, ops, FLAGS.output+'.pickle', FLAGS.output) # 0.1s
        time2 = time.time()
        print('time: ', time2 - time1)

        mlab.clf(fig)
        img, _ = dataset.get_image(i)
        pc = dataset.get_lidar(i)[:, 0:3]
        show_lidar(pc, calibs, fig, img_fov=True, img_width=img.shape[1], img_height=img.shape[0])

        box3d_pts_3d_velo_list = []
        for box_3d in box_3d_list:
            box3d_pts_3d_velo = calibs.project_rect_to_velo(box_3d)
            box3d_pts_3d_velo_list.append(box3d_pts_3d_velo)
        draw_gt_boxes3d(box3d_pts_3d_velo_list, fig)
        '''
        img, _ = dataset.get_image(i)
        print('img: ', img.shape)
        pc = dataset.get_lidar(i)[:, 0:3]
        cv2.imshow('0', img)
        #show_lidar(pc, calibs, fig, img_fov = False, img_width = img.shape[1], img_height = img.shape[0])
        #show_lidar_on_image(pc, img, calibs, img_width=img.shape[1], img_height=img.shape[0])
        #cv2.waitKey(1)
        class_IDs, scores, bounding_boxs = get_2d_box_yolo(img, net)
        print('shape: ', class_IDs.shape, scores.shape, bounding_boxs.shape)
        '''
        if i % 10 == 0:
            input()
    input()

if __name__ == '__main__':
    print('start')
    demo()
