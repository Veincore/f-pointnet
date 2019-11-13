from gluoncv import model_zoo, data, utils
from matplotlib import pyplot as plt
import numpy as np
import cv2
import mxnet as mx

net = model_zoo.get_model('yolo3_darknet53_voc', pretrained=True)

im_fname = utils.download('https://raw.githubusercontent.com/zhreshold/' +
                          'mxnet-ssd/master/data/demo/dog.jpg',
                          path='dog.jpg')
img = cv2.imread('dog.jpg')
img = mx.nd.array(img[:, :, ::-1])
#im_fname = 'D:\\Detectron_Data\\2011_09_26_drive_0001_sync\\image_02\\data\\0000000000.png'
#x, img = data.transforms.presets.yolo.load_test(im_fname, short = 512)
x, img = data.transforms.presets.yolo.transform_test(img, short = 512)
print('Shape of pre-processed image:', x.shape)

'''
class_IDS:
   0        1      2    3     4     5   6   7   8     9      10
aeroplane bicycle bird boat bottle bus car cat chair cow diningtable 
 11   12      13       14       15       16    17   18      19
 dog horse motorbike person pottedplant sheep sofa train tvmonitor 
'''
class_IDs, scores, bounding_boxs = net(x)

print('class_id: ',     class_IDs)  # NDArray [batch, 100, 1]
print('scores: ', scores)  # [batch, 100, 1]
print('box: ', bounding_boxs)  # [batch, 100, 4]

ax = utils.viz.plot_bbox(img, bounding_boxs[0], scores[0],
                         class_IDs[0], class_names=net.classes)
plt.show()
