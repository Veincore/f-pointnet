from gluoncv import model_zoo, data, utils
from matplotlib import pyplot as plt
import numpy as np
import cv2
import mxnet as mx

def show_image_with_2d_boxes(img, box_list):
    for box in box_list:
        cv2.rectangle(img, (int(box[0]),int(box[1])),
            (int(box[2]),int(box[3])), (0,255,0), 2)
    cv2.imshow('0', img)

def transform_bbox_inverse(bbox_lists, img_ori_shape, img_shape):
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

net = model_zoo.get_model('yolo3_darknet53_voc', pretrained=True)

im_fname = utils.download('https://raw.githubusercontent.com/zhreshold/' +
                          'mxnet-ssd/master/data/demo/dog.jpg',
                          path='dog.jpg')
img_ori = cv2.imread('dog.jpg')
img = img_ori
img = mx.nd.array(img[:, :, ::-1])
x, img = data.transforms.presets.yolo.transform_test(img, short = 512)
'''
im_fname = 'D:\\Detectron_Data\\2011_09_26_drive_0001_sync\\image_02\\data\\0000000080.png'
x, img = data.transforms.presets.yolo.load_test(im_fname, short = 512)
'''
print('Shape of pre-processed image:', x.shape)

'''
class_IDS:
   0        1      2    3     4     5   6   7   8     9      10
aeroplane bicycle bird boat bottle bus car cat chair cow diningtable 
 11   12      13       14       15       16    17   18      19
 dog horse motorbike person pottedplant sheep sofa train tvmonitor 
'''
class_IDs, scores, bounding_boxs = net(x)
class_IDs, scores, bounding_boxs = class_IDs.asnumpy(), scores.asnumpy(), bounding_boxs.asnumpy()
class_id_index = np.where(class_IDs > -1)
class_IDs = class_IDs[class_id_index]
scores = scores[class_id_index]
bounding_boxs = bounding_boxs[:, :len(class_IDs), :].squeeze(0)

class_id_index = [i for i, e in enumerate(class_IDs) if e in [6, 14, 1]]
class_IDs = class_IDs[class_id_index]
scores = scores[class_id_index]
bounding_boxs = bounding_boxs[class_id_index, :]

print(img_ori.shape)
bounding_boxs = transform_bbox_inverse(bounding_boxs, img_ori.shape, img.shape)
print(bounding_boxs)
show_image_with_2d_boxes(img_ori, bounding_boxs)
cv2.waitKey(0)
'''
print('class_id: ',     class_IDs)  # NDArray [batch, 100, 1]
print('scores: ', scores)  # [batch, 100, 1]
print('box: ', bounding_boxs)  # [batch, 100, 4]
print('id_index: ', class_id_index)

ax = utils.viz.plot_bbox(img, bounding_boxs[0], scores[0],
                         class_IDs[0], class_names=net.classes)
plt.show()
'''