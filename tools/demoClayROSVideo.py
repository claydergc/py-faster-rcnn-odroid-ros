#!/usr/bin/env python

import roslib; roslib.load_manifest('tools')

import time
import rospy
import math
import numpy as np
import sys
import cv2

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer

caffe_root = '../caffe-fast-rcnn/'
sys.path.insert(0, caffe_root + 'python')

import caffe, os, sys, cv2
import argparse

CLASSES = ('__background__',
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
         'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}

net = 0

def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return	
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]      
        cv2.rectangle(im, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), (0,0,255), 2) 
        cv2.putText(im, class_name + ':' +str(score), (int(bbox[0]),int(bbox[1] - 2)), font, 0.4, (0,0,255), 1)
        #print class_name,':', score
    
    #return inds
		
    #cv2.imshow('img', im)

def showResults(im, class_name, dets, thresh=0.5):    

    inds = np.where(dets[:, -1] >= thresh)[0]

    if len(inds) == 0:
        return

    for i in inds:
        score = dets[i, -1] 
        print class_name,':', score

    return score
		

def demo(net, im):
    """Detect object classes in an image using pre-computed object proposals."""

    detsList = []
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes, originalShape, resizedShape = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    #print 'SHAPE', im.shape

    # Visualize detections for each class
    #CONF_THRESH = 0.7
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3

    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        i = vis_detections(im, cls, dets, thresh=CONF_THRESH)
        #inds.append(i)
        detsList.append(dets)
        #score = showResults(im, cls, dets, thresh=CONF_THRESH)
    return detsList
        
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

    args = parser.parse_args()

    return args

count = 0
frameSkip = 1
detsList = []

class image_converter:

	def __init__(self):    
	    self.bridge = CvBridge()
	    self.image_sub = rospy.Subscriber("camera/image_raw",Image,self.callback,queue_size = 1)

	def callback(self,data):

		global net
		global count
		global frameSkip
		global detsList

		try:
			cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
		except CvBridgeError as e:
			print(e)

	    #(rows,cols,channels) = cv_image.shape
	    #if cols > 60 and rows > 60 :
	    #  cv2.circle(cv_image, (50,50), 10, 255)                        

		if count%frameSkip==0:
			detsList = demo(net, cv_image)
			count = 0
		elif detsList:
			for cls_ind, cls in enumerate(CLASSES[1:]):
				dets = detsList[cls_ind]
				vis_detections(cv_image, cls, dets, 0.8)

		cv2.imshow("Video", cv_image)
		cv2.waitKey(3)
		count = count + 1

	    #if cv2.waitKey(33) == ord('a'):
   	    #	print "pressed a"
		#rospy.sleep(0.5)
            	#demo(net, cv_image)
		#rospy.sleep(0.5)
	    
if __name__ == '__main__':
	
    global net

    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                              NETS[args.demo_net][1])
	
    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    caffe.set_mode_cpu()
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)    

    #im_names = ['000456.jpg', '000542.jpg', '001150.jpg',
    #            '001763.jpg', '004545.jpg']

    
    ic = image_converter()
    rospy.init_node('demoClayROSVideo', anonymous=True)
    try:
      rospy.spin()
    except KeyboardInterrupt:
      print("Shutting down")
    cv2.destroyAllWindows()

    #for im_name in im_names:
    #    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    #    print 'Demo for data/demo/{}'.format(im_name)
    #    demo(net, im_name)

    #while not rospy.is_shutdown() and False:
    #	R.rate.sleep()
    sys.exit()
        
    #resultados.close()
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

