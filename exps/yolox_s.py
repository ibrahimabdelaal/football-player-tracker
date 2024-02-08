#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
       
        self.num_classes = 4
        self.depth = 1.33
        self.width = 1.25
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.data_dir = "/home/dell/ByteTrack_HOME/yolo_x_train/YOLOX/datasets/coco_dataset"
        self.train_ann = "/home/dell/ByteTrack_HOME/yolo_x_train/YOLOX/datasets/coco_dataset/annotion/instances_train2017.json"
        self.val_ann = "/home/dell/ByteTrack_HOME/yolo_x_train/YOLOX/datasets/coco_dataset/annotion/instances_val2017.json"   # change to train.json when running on training set
        # self.inference_with_ground_truth = True
        self.test_ann = "/home/dell/ByteTrack_HOME/yolo_x_train/YOLOX/datasets/coco_dataset/annotion/instances_test2017.json"
        
        # if self.inference_with_ground_truth:
        # self.input_size = (640, 640)
        # self.test_size = (640, 640)      
        # # else:
        self.input_size = (1280, 1280)
       # self.test_size = (1280, 1280)

        self.test_size = (1280, 1280)
        self.random_size = None
        self.max_epoch =15
        self.print_interval = 5
        self.eval_interval = 5
        self.test_conf = 0.4
        self.nmsthre = 0.35
        self.no_aug_epochs = 10
        self.basic_lr_per_img = 0.001 / 64.0
        self.warmup_epochs = 1
