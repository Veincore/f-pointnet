import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # '../frustum-pointnets/infer'
ROOT_DIR = os.path.dirname(BASE_DIR)  # '../frustum-pointnets'
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'mayavi'))

if(os.path.exists('/opt/ros/kinetic/lib/python2.7/dist-packages/')):
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import kitti.kitti_util as utils

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
        self.num_samples = 108

        self.image_dir = os.path.join(self.root_dir, 'image_02/data')
        self.lidar_dir = os.path.join(self.root_dir, 'velodyne_points/data')
        self.calib_dir = os.path.join(self.root_dir, '2011_09_26_calib/2011_09_26')
        #self.image_dir = os.path.join(self.root_dir, 'image_02\\data')
        #self.calib_dir = os.path.join(self.root_dir, '2011_09_26_calib\\2011_09_26')
        #self.lidar_dir = os.path.join(self.root_dir, 'velodyne_points\\data')

    def __len__(self):
        return self.num_samples

    def get_image(self, idx):
        assert(idx < self.num_samples)
        img_filename = os.path.join(self.image_dir, '%010d.png'%(idx))
        #img_filename = os.path.join(self.image_dir, '0000000000.png')
        print('filename: ', img_filename)
        return utils.load_image(img_filename)

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
    from viz_util import draw_lidar_simple, draw_lidar, draw_gt_boxes3d

    mlab.clf(fig)
    print(('All point num: ', pc_velo.shape[0]))
    #fig = mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(1000, 500))
    if img_fov:
        pc_velo = get_lidar_in_image_fov(pc_velo, calib, 0, 0,
            img_width, img_height)
        print(('FOV point num: ', pc_velo.shape[0]))
    draw_lidar(pc_velo, fig=fig)
    mlab.show(1)

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
    cv2.imshow('0', img)
    return img

def demo():
    if 'mlab' not in sys.modules: import mayavi.mlab as mlab

    dataset = kitti_object_infer('/media/vdc/backup/database_backup/Chris/f-pointnet/2011_09_26_drive_0001_sync')
    calibs = calib_infer('/media/vdc/backup/database_backup/Chris/f-pointnet/2011_09_26_drive_0001_sync/2011_09_26_calib/2011_09_26')
    #dataset = kitti_object_infer('D:\\Detectron_Data\\2011_09_26_drive_0001_sync')
    #calibs = calib_infer('D:\\Detectron_Data\\2011_09_26_drive_0001_sync\\2011_09_26_calib\\2011_09_26')
    fig = mlab.figure(figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1000, 500))
    for i in range(len(dataset)):
        img = dataset.get_image(i)
        print('img: ', img.shape)
        pc = dataset.get_lidar(i)[:, 0:3]
        #cv2.imshow('0', img)
        show_lidar(pc, calibs, fig, img_fov = False, img_width = img.shape[1], img_height = img.shape[0])
        show_lidar_on_image(pc, img, calibs, img_width=img.shape[1], img_height=img.shape[0])
        cv2.waitKey(1)

if __name__ == '__main__':
    print('start')
    demo()
