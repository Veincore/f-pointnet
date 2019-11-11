import os
import sys

import mayavi as mlab

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # '../frustum-pointnets/infer'
ROOT_DIR = os.path.dirname(BASE_DIR)  # '../frustum-pointnets'
sys.path.append(ROOT_DIR)

import cv2
import numpy as np
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
        self.V2C = np.zeros([4, 4])
        self.V2C[:3, :3] = np.reshape(calibs['R'], [3, 3])
        self.V2C[:3, 3:4] = np.reshape(calibs['T'], [3, 1])
        self.V2C[3, 3] = 1


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







class kitti_object_infer():
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.num_samples = 108

        #self.image_dir = os.path.join(self.root_dir, 'image_02/data')
        #self.lidar_dir = os.path.join(self.root_dir, 'velodyne_points/data')
        #self.calib_dir = os.path.join(self.root_dir, '2011_09_26_calib/2011_09_26')
        self.image_dir = os.path.join(self.root_dir, 'image_02\\data')
        self.calib_dir = os.path.join(self.root_dir, '2011_09_26_calib\\2011_09_26')
        self.lidar_dir = os.path.join(self.root_dir, 'velodyne_points\\data')


    def __len__(self):
        return self.num_samples

    def get_image(self, idx):
        assert(idx < self.num_samples)
        #img_filename = os.path.join(self.image_dir, '%10d.png'%(idx))
        img_filename = os.path.join(self.image_dir, '0000000000.png')
        print('filename: ', img_filename)
        return utils.load_image(img_filename)

    def get_lidar(self, idx):
        assert(idx < self.num_samples)
        #lidar_filename = os.path.join(self.lidar_dir, '%06d.bin'%(idx))
        lidar_filename = os.path.join(self.lidar_dir, '0000000000.bin')
        return utils.load_velo_scan(lidar_filename)

    def get_calibration(self):
        return calib_infer(self.calib_dir)


'''
def demo():
    raw_input
'''

if __name__ == '__main__':
    print('start')
    data = kitti_object_infer('D:\\Detectron_Data\\2011_09_26_drive_0001_sync')
    #img = data.get_image(0)
    #cv2.imshow('prep', img)
    #cv2.waitKey(0)
    calibs = calib_infer('D:\\Detectron_Data\\2011_09_26_drive_0001_sync\\2011_09_26_calib\\2011_09_26')