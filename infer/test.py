import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


if __name__ == '__main__':
    print('ok', BASE_DIR)



'''
from gluoncv import model_zoo, data, utils
from matplotlib import pyplot as plt
import time

net = model_zoo.get_model('yolo3_darknet53_voc', pretrained=True)

im_fname = utils.download('https://raw.githubusercontent.com/zhreshold/' +
                          'mxnet-ssd/master/data/demo/dog.jpg',
                          path='dog.jpg')
x, img = data.transforms.presets.yolo.load_test(im_fname, short = 512)
print('Shape of pre-processed image:', x.shape)

time1 = time.time()
class_IDs, scores, bounding_boxs = net(x)
time2 = time.time()
print('time: ', time2 - time1)

ax = utils.viz.plot_bbox(img, bounding_boxs[0], scores[0],
                         class_IDs[0], class_names=net.classes)
plt.show()
'''