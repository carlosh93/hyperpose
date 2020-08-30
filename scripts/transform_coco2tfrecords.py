#!/usr/bin/env python3

import os
import cv2
import sys
import math
import json
import time
import argparse
import matplotlib
import multiprocessing
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from hyperpose import Config, Model, Dataset
from pathlib import Path
import shutil

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FastPose.')
    parser.add_argument("--model_type",
                        type=str,
                        default="Openpose",
                        help="human pose estimation model type, available options: Openpose, LightweightOpenpose ,MobilenetThinOpenpose, PoseProposal")
    parser.add_argument("--dataset_type",
                        type=str,
                        default="MSCOCO",
                        help="dataset name,to determine which dataset to use, available options: MSCOCO, MPII ")
    parser.add_argument("--dataset_path",
                        type=str,
                        default="/data/",
                        help="dataset path,to determine the path to load the dataset")
    parser.add_argument('--train_type',
                        type=str,
                        default="Single_train",
                        help='train type, available options: Single_train, Parallel_train')

    args = parser.parse_args()
    # config model
    Config.set_model_type(Config.MODEL[args.model_type])
    # config train type
    Config.set_train_type(Config.TRAIN[args.train_type])
    # config dataset
    Config.set_dataset_type(Config.DATA[args.dataset_type])
    Config.set_dataset_path(args.dataset_path)

    config = Config.get_config()
    dataset = Dataset.get_dataset(config)
    train_dataset = dataset.save_train_tfrecord_dataset()
    Path("../data").mkdir(parents=True, exist_ok=True)
    shutil.move('coco_pose_data.tfrecord', '../data/')
