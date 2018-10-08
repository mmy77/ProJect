import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

from segs.deeplab_v3_plus import _build_deeplab
from segs.common_configure import jupyter_flag
from tf_utils import dense_crf
import os
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from matplotlib import gridspec
from scipy import misc
from time import time
import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

LABEL_NAMES = np.asarray([
    'background','person','sky','plant','Grass','flower','car','cat','dog','sea','table',
    'building','unknown'#,'ignore_label'
])

def create_pascal_label_colormap():
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.

    Returns:
    A Colormap for visualizing segmentation results.
    """
    """colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3
    """
    colormap = [[0, 0, 0],
                [192, 128, 128],
                [0, 191, 255],
                [0, 128, 0],
                [0, 255, 0],
                [255, 0, 0],
                [128, 128, 128],
                [64, 0, 0],
                [255, 215, 0],
                [0, 0, 255],
                [160, 32, 240],
                [0, 139, 139],
                [255, 255, 0]]
                
    """
    colormap = [[0, 0, 0],
                [255, 255, 255],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0,0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]]
    """
   # colormap = [0,255,0,0,0,0,0,0,0,0,0,0,0]
    colormap = np.array(colormap)
    return colormap


def label_to_color_image(label):
    """Adds color defined by the dataset colormap to the label.

    Args:
    label: A 2D array with integer type, storing the segmentation label.

    Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

    Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
    """
    if label.ndim != 2:
        print(label.ndim)
        raise ValueError('Expect 2-D input label')

    colormap = create_pascal_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]


FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)


def vis_segmentation(image, seg_map):
    """Visualizes input image, segmentation map and overlay view."""
    #plt.figure(figsize=(15, 5))
    #grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])
    
    plt.figure(figsize=(15, 5))
    grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

    plt.subplot(grid_spec[0])
    plt.imshow(image)
    plt.axis('off')
    plt.title('input image')

    plt.subplot(grid_spec[1])
    seg_image = label_to_color_image(seg_map).astype(np.uint8)
    plt.imshow(seg_image)
    plt.axis('off')
    plt.title('segmentation map')

    plt.subplot(grid_spec[2])
    plt.imshow(image)
    plt.imshow(seg_image, alpha=0.7)
    plt.axis('off')
    plt.title('segmentation overlay')
    # segmap1
    

    unique_labels = np.unique(seg_map)
    ax = plt.subplot(grid_spec[3])
    plt.imshow(
      FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
    ax.yaxis.tick_right()
    plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
    plt.xticks([], [])
    ax.tick_params(width=0.0)
    plt.grid('off')
    plt.show()
    return seg_image

def vis_segmentation_orig(image, seg_map):
    """Visualizes input image, segmentation map and overlay view."""
    #plt.figure(figsize=(15, 5))
    #grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])
    
    plt.figure(figsize=(15, 5))
    grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

    plt.subplot(grid_spec[0])
    plt.imshow(image)
    plt.axis('off')
    plt.title('input image')

    plt.subplot(grid_spec[1])
    seg_image = label_to_color_image(seg_map).astype(np.uint8)
    plt.imshow(seg_image)
    plt.axis('off')
    plt.title('segmentation_orig')

    plt.subplot(grid_spec[2])
    plt.imshow(image)
    plt.imshow(seg_image, alpha=0.7)
    plt.axis('off')
    plt.title('segmentation_orig_overlay')
    # segmap1
    

    unique_labels = np.unique(seg_map)
    ax = plt.subplot(grid_spec[3])
    plt.imshow(
      FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
    ax.yaxis.tick_right()
    plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
    plt.xticks([], [])
    ax.tick_params(width=0.0)
    plt.grid('off')
    plt.show()
    return seg_image


def inference(sess, image):
    image_shape = image.size
    
    # segment compare
    resized_image = image.convert('RGB').resize(net_shape, Image.ANTIALIAS)
    #resized_seg_map = sess.run(predictions, feed_dict={inputs: np.expand_dims(np.asarray(resized_image), axis=0)})
    
    # end
    logits = outputs_to_scales_to_logits['semantic']['merged_logits']
    logits = tf.image.resize_nearest_neighbor(logits, net_shape, align_corners=True)
    
    # CRF
    logits_softmax = tf.nn.softmax(logits)
    raw_output_up = tf.py_func(dense_crf, [logits_softmax, tf.expand_dims(resized_image, dim=0)], tf.float32)
    raw_output_up = tf.argmax(raw_output_up, axis=-1)
    pred = tf.expand_dims(raw_output_up, dim=3)
    pred = sess.run(pred, feed_dict={inputs: np.expand_dims(np.asarray(resized_image), axis=0)})
    resized_seg_map = np.uint8(np.squeeze(pred))

    #     resized_seg_map = np.uint8(np.squeeze(resized_seg_map))

    resized_seg_map = Image.fromarray(resized_seg_map, mode='P')
    seg_map = resized_seg_map.resize(list(image_shape), Image.NEAREST)
    seg_map = np.asarray(seg_map)
    return image, seg_map


def inference_orig(sess, image):
    image_shape = image.size
    
    # segment compare
    resized_image = image.convert('RGB').resize(net_shape, Image.ANTIALIAS)
    resized_seg_map = sess.run(predictions, feed_dict={inputs: np.expand_dims(np.asarray(resized_image), axis=0)})
    
    # end
    #logits = outputs_to_scales_to_logits['semantic']['merged_logits']
    #logits = tf.image.resize_nearest_neighbor(logits, net_shape, align_corners=True)
    
    
    resized_seg_map = np.uint8(np.squeeze(resized_seg_map))
    resized_seg_map = Image.fromarray(resized_seg_map, mode='P')
    seg_map = resized_seg_map.resize(list(image_shape), Image.NEAREST)
    seg_map = np.asarray(seg_map)
    
    return image, seg_map
class filedialogdemo(QWidget):

    def __init__(self, parent=None):
        super(filedialogdemo, self).__init__(parent)
        layout = QGridLayout()

        layout.setSpacing(30)

        self.btn = QPushButton()
        self.btn.clicked.connect(self.loadFile)
        self.btn.setText("get pic")
        layout.addWidget(self.btn,1,1)

        self.label_filename = QLabel()#
        layout.addWidget(self.label_filename,1,3,1,10)
        
        self.label = QLabel()
        layout.addWidget(self.label,2,1,12,12)
        self.label.setAutoFillBackground(True)
        palette = QPalette()
        palette.setColor(QPalette.Window,Qt.blue)
        self.label.setPalette(palette)

        self.btn_3 = QPushButton()
        self.btn_3.clicked.connect(self.compute_pic)
        self.btn_3.setText("compute")
        layout.addWidget(self.btn_3,8,13,2,2)

        self.label_2 = QLabel()
        layout.addWidget(self.label_2,2,15,12,12)
        self.label_2.setAutoFillBackground(True)
        self.label_2.setPalette(palette)

        #layout.addWidget(layout_pic)
        self.btn_2 = QPushButton()
        self.btn_2.clicked.connect(self.load_file)
        self.btn_2.setText("get file")
        layout.addWidget(self.btn_2,15,1)

        self.label_filename2_1 = QLabel()
        layout.addWidget(self.label_filename2_1,15,3,1,10)

        self.btn_2_2 = QPushButton()
        self.btn_2_2.clicked.connect(self.save_file)
        self.btn_2_2.setText("save file")
        layout.addWidget(self.btn_2_2,15,15)

        self.label_filename2_2 = QLabel()
        layout.addWidget(self.label_filename2_2,15,18,1,10)


        self.label_low = QLabel() #2-1 img
        layout.addWidget(self.label_low,16,1,12,12)
        self.label_low.setAutoFillBackground(True)
        self.label_low.setPalette(palette)

        self.btn_3_low = QPushButton()
        self.btn_3_low.clicked.connect(self.compute_file)
        self.btn_3_low.setText("compute")
        layout.addWidget(self.btn_3_low,20,13,2,2)

        self.label_2_low = QLabel() #2-2 img
        layout.addWidget(self.label_2_low,16,15,12,12)
        self.label_2_low.setAutoFillBackground(True)
        self.label_2_low.setPalette(palette)

        self.status = QProgressBar()
        layout.addWidget(self.status,28,1,2,10)

        self.status_text = QLabel()
        layout.addWidget(self.status_text,28,12,2,10)

        self.setWindowTitle("test")

        self.setLayout(layout)
        self.move(600,600)

    def loadFile(self):
        print("load--file")
        self.fname, _ = QFileDialog.getOpenFileName(self, 'choose', '/home/lxq/code_huawei/', 'Image files(*.jpg *.gif *.png)')
        pixmap = QPixmap()
        pixmap.load(self.fname)
        pixmap = pixmap.scaledToWidth(240)
        self.label.setPixmap(pixmap)
       # self.label.setPixmap(QPixmap(self.fname))
        self.label.setScaledContents(True)

    def compute_pic(self):
        imgpath = self.fname
        start = time()
        image = Image.open(imgpath)#+ '.jpg'))
        
        #image = Image.open(os.path.join(IMAGE_DIR, name))
        _, seg_map = inference(sess, image)
        _,seg_map1 = inference_orig(sess,image) 
        stop = time()
        print('time using:', stop-start)
        #np.save(os.path.join(SAVE_DIR, name + '.npy'), seg_map)
        seg_mapm = label_to_color_image(seg_map).astype(np.uint8)
       
        misc.imsave(imgpath[:-4]+'_seg.png',seg_mapm)
        resultpath = imgpath[:-4]+'_seg.png'
        pixmap = QPixmap()
        pixmap.load(resultpath)
        pixmap = pixmap.scaledToWidth(240)
        self.label_2.setPixmap(pixmap)
       # self.label.setPixmap(QPixmap(self.fname))
        self.label_2.setScaledContents(True)
        #     if idx % 50 == 0 and idx != 0:
        #         print('{:.2f} %'.format(100 * idx/len(pic_names)))
        #seg_image = vis_segmentation(image, seg_map)
        #seg_image1 = vis_segmentation_orig(image, seg_map1)

    def load_file(self):
        print("load--file")
        self.filename = QFileDialog.getExistingDirectory(self,'choose file ',"./")
        self.label_filename2_1.setText(self.filename)
    def save_file(self):
        self.sfilename= QFileDialog.getExistingDirectory(self,'choose file ',"./")
        self.label_filename2_2.setText(self.sfilename)
    def compute_file(self):
        imgfile = self.filename
        print(imgfile)
        savefile = self.sfilename
        print(savefile)
        imgnames = os.listdir(imgfile)
        global count
        count  = 0
        global countall
        countall = len(imgnames)
        self.status.setMaximum(countall)
        self.status.setMinimum(0)
        for name in imgnames:
            start = time()
            image = Image.open(imgfile+'/'+name )#+ '.jpg'))
            print(name)
            self.status_text.setText(imgfile+'/'+name)

            pixmap = QPixmap()
            pixmap.load(imgfile+'/'+name)
            pixmap = pixmap.scaledToWidth(240)
            self.label_low.setPixmap(pixmap)
           # self.label.setPixmap(QPixmap(self.fname))
            self.label_low.setScaledContents(True)

            #image = Image.open(os.path.join(IMAGE_DIR, name))
            _, seg_map = inference(sess, image)
            _,seg_map1 = inference_orig(sess,image) 
            stop = time()
            print('time using:', stop-start)
            #np.save(os.path.join(SAVE_DIR, name + '.npy'), seg_map)
            seg_mapm = label_to_color_image(seg_map).astype(np.uint8)
            name = name.replace('/','')
            misc.imsave(os.path.join(savefile,name),seg_mapm)

            pixmap = QPixmap()
            pixmap.load(os.path.join(savefile,name))
            pixmap = pixmap.scaledToWidth(240)
            self.label_2_low.setPixmap(pixmap)
           # self.label.setPixmap(QPixmap(self.fname))
            self.label_2_low.setScaledContents(True)

            count = count+1
            self.status.setValue(count)

FLAGS = jupyter_flag()
net_shape = [513, 513]
with tf.device('/GPU:0'):

    inputs = tf.placeholder(name='inputs', shape=[1, 513, 513, 3], dtype=tf.float32)
    outputs_to_scales_to_logits = _build_deeplab(inputs, None, [], FLAGS, is_training=False)
    merged_logits = tf.expand_dims(
        tf.argmax(outputs_to_scales_to_logits['semantic']['merged_logits'], axis=-1),
        axis=-1)
    predictions = tf.image.resize_nearest_neighbor(merged_logits, net_shape, align_corners=True)

config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
sess = tf.Session(config=config)
saver = tf.train.Saver()
saver.restore(sess, save_path='/home/lxq/code_huawei/model_deeplabv3+/model.ckpt')
if __name__ == '__main__':
    app = QApplication(sys.argv)
    fileload =  filedialogdemo()
    fileload.show()
    sys.exit(app.exec_())


"""LABEL_NAMES = np.asarray([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
])
"""





        #     if idx % 50 == 0 and idx != 0:
        #         print('{:.2f} %'.format(100 * idx/len(pic_names)))
        #seg_image = vis_segmentation(image, seg_map)
        #seg_image1 = vis_segmentation_orig(image, seg_map1)