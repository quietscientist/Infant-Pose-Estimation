# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

# Modified by Xiaofei Huang (xhuang@ece.neu.edu) and Nihang Fu (nihang@ece.neu.edu)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
import os.path as osp

import json

from collections import namedtuple

import cv2
import numpy as np

import torch
from torch.utils.data import Dataset


from utils import smpl_to_openpose

Keypoints = namedtuple('Keypoints',
                       ['keypoints', 'gender_gt', 'gender_pd', 'dims'])

Keypoints.__new__.__defaults__ = (None,) * len(Keypoints._fields)
# COCO limb sequence (0-based indexing)
coco_limb_sequence = [
    (0, 1),  # Nose to Left Eye
    (0, 2),  # Nose to Right Eye
    (1, 3),  # Left Eye to Left Ear
    (2, 4),  # Right Eye to Right Ear
    (5, 7),  # Left Shoulder to Left Elbow
    (7, 9),  # Left Elbow to Left Wrist
    (6, 8),  # Right Shoulder to Right Elbow
    (8, 10), # Right Elbow to Right Wrist
    (5, 6),  # Left Shoulder to Right Shoulder
    (5, 11), # Left Shoulder to Left Hip
    (6, 12), # Right Shoulder to Right Hip
    (11, 12),# Left Hip to Right Hip
    (11, 13),# Left Hip to Left Knee
    (13, 15),# Left Knee to Left Ankle
    (12, 14),# Right Hip to Right Knee
    (14, 16) # Right Knee to Right Ankle
]

limb_sequence = [
    (0,15),
    (0,16),
    (15,16),
    (15,17),
    (16,18),
    (1,2),
    (2,3),
    (3,4),
    (1,5),
    (5,6),
    (6,7),
    (2,9),
    (5,12),
    (8,9),
    (9,10),
    (10,11),
    (8,12),
    (12,13),
    (13,14)
    ]

mapping = {0:0,1:15,2:16,3:17,4:18,5:2,6:5,7:3,8:6,9:4,10:7,11:9,12:12,13:10,14:13,15:11,16:14}

# COCO part list
part_list = {
    0: "Nose",
    1: "Left Eye",
    2: "Right Eye",
    3: "Left Ear",
    4: "Right Ear",
    5: "Left Shoulder",
    6: "Right Shoulder",
    7: "Left Elbow",
    8: "Right Elbow",
    9: "Left Wrist",
    10: "Right Wrist",
    11: "Left Hip",
    12: "Right Hip",
    13: "Left Knee",
    14: "Right Knee",
    15: "Left Ankle",
    16: "Right Ankle"
}

def reorder_keypoints(keypoints, confidence_scores):
    """
    Reorder the keypoints to the OpenPose format.
    The OpenPose format is as follows:
    0-17: [nose, neck, right_shoulder, right_elbow, right_wrist, left_shoulder, left_elbow, left_wrist,
           right_hip, right_knee, right_ankle, left_hip, left_knee, left_ankle, right_eye, left_eye, right_ear, left_ear]
    The input 'keypoints' is a list of (x, y, c) tuples, where c is the confidence score.
    """

    # Reorder the keypoints to the OpenPose format
    keypoints = [keypoints[i] for i in [0, 17, 6, 8, 10, 5, 7, 9, 18, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]]
    confidence_scores = [confidence_scores[i] for i in [0, 17, 6, 8, 10, 5, 7, 9, 18, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]]

    return keypoints, confidence_scores

def rescale_keypoints(keypoints, scale):
    """
    Rescale the keypoints by the given scale.
    The input 'keypoints' is a list of (x, y) tuples
    """

    # Rescale the keypoints
    keypoints = [(x * scale, y * scale) for (x, y) in keypoints]

    return keypoints


def convert_coco_to_openpose(coco_keypoints, confidence_scores):
    """
    Convert COCO keypoints to OpenPose keypoints with the neck keypoint as the midpoint between the two shoulders.
    COCO keypoints format (17 keypoints): [nose, left_eye, right_eye, left_ear, right_ear,
                                           left_shoulder, right_shoulder, left_elbow, right_elbow,
                                           left_wrist, right_wrist, left_hip, right_hip,
                                           left_knee, right_knee, left_ankle, right_ankle]
    OpenPose keypoints format (18 keypoints): COCO keypoints + [neck]
    The neck is not a part of COCO keypoints and is computed as the midpoint between the left and right shoulders.
    """

    # Assuming coco_keypoints is a list of (x, y) tuples
    nose, left_eye, right_eye, left_ear, right_ear, \
    left_shoulder, right_shoulder, left_elbow, right_elbow, \
    left_wrist, right_wrist, left_hip, right_hip, \
    left_knee, right_knee, left_ankle, right_ankle = coco_keypoints

    # Calculate the neck as the midpoint between left_shoulder and right_shoulder
    neck_x = (left_shoulder[0] + right_shoulder[0]) / 2
    neck_y = (left_shoulder[1] + right_shoulder[1]) / 2
    neck = (neck_x, neck_y)

    mid_x = (left_hip[0] + right_hip[0]) / 2
    mid_y = (left_hip[1] + right_hip[1]) / 2
    mid = (mid_x, mid_y)

    # Assuming coco_keypoints is a list of (x, y) tuples
    c_nose, c_left_eye, c_right_eye, c_left_ear, c_right_ear, \
    c_left_shoulder, c_right_shoulder, c_left_elbow, c_right_elbow, \
    c_left_wrist, c_right_wrist, c_left_hip, c_right_hip, \
    c_left_knee, c_right_knee, c_left_ankle, c_right_ankle = confidence_scores

    # Calculate the neck as the midpoint between left_shoulder and right_shoulder
    c_neck = (c_left_shoulder + c_right_shoulder) / 2
    c_mid = (c_left_hip + c_right_hip) / 2

    # Construct the OpenPose keypoints including the neck
    openpose_keypoints = [
        nose, left_eye, right_eye, left_ear, right_ear,
        left_shoulder, right_shoulder, left_elbow, right_elbow,
        left_wrist, right_wrist, left_hip, right_hip,
        left_knee, right_knee, left_ankle, right_ankle,
        neck, mid  # Adding the neck as the last keypoint
    ]
    
    openpose_confidences = [
        c_nose, c_left_eye, c_right_eye, c_left_ear, c_right_ear,
        c_left_shoulder, c_right_shoulder, c_left_elbow, c_right_elbow,
        c_left_wrist, c_right_wrist, c_left_hip, c_right_hip,
        c_left_knee, c_right_knee, c_left_ankle, c_right_ankle,
        c_neck, c_mid  # Adding the neck as the last keypoint
    ]

    openpose_keypoints, confidences = reorder_keypoints(openpose_keypoints, openpose_confidences)
    openpose_keypoints = rescale_keypoints(openpose_keypoints, 1)

    return openpose_keypoints, confidences

def create_dataset(dataset='openpose', data_folder='data', **kwargs):
    if dataset.lower() == 'openpose':
        return OpenPose(data_folder, **kwargs)
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))


def read_keypoints(keypoint_fn, use_hands=True, use_face=True,
                   use_face_contour=False):



####################
### READ KEYPOINTS #  
####################
    with open(keypoint_fn) as keypoint_file:
        data = json.load(keypoint_file)
    
    keypoints = []
    dims = []

    try:
        body_keypoints = np.array(data['annotations'], dtype=np.float32)
        H = 720
        W = 1280
        
        body_keypoints = body_keypoints.reshape([-1, 3])
        keypoints.append(body_keypoints)
        dims.append([H,W])

    except: 
        n_frames = len(data)

        for frame in range(n_frames):
            data_1, data_2 = convert_coco_to_openpose(data[frame]['instances'][0]['keypoints'], data[frame]['instances'][0]['keypoint_scores'])
            _,_,W,H = data[frame]['instances'][0]['bbox'][0]
            
            data_x = []
            for e, kp in enumerate(data_1):
                data_x.append([data_1[e][0],data_1[e][1],data_2[e]])

            body_keypoints = np.array(data_x).flatten()
            body_keypoints = body_keypoints.reshape([-1, 3])
    
            keypoints.append(body_keypoints)
            dims.append([H,W])

        """        data_1, data_2 = convert_coco_to_openpose(data[10]['instances'][0]['keypoints'], data[10]['instances'][0]['keypoint_scores'])
                _,_,W,H = data[10]['instances'][0]['bbox'][0]
                print(W, H)

                data_x = []
                for e, kp in enumerate(data_1):
                    data_x.append([data_1[e][0],data_1[e][1],data_2[e]])

                body_keypoints = np.array(data_x).flatten()
        """


    gender_pd = []
    gender_gt = []
    
    return Keypoints(keypoints=keypoints, gender_pd=gender_pd,
                     gender_gt=gender_gt, dims=dims)


class OpenPose(Dataset):

    NUM_BODY_JOINTS = 19
    NUM_HAND_JOINTS = 20

    def __init__(self, data_folder,
                 keyp_folder='keypoints',
                 use_hands=False,
                 use_face=False,
                 dtype=torch.float32,
                 model_type='smplx',
                 joints_to_ign=None,
                 use_face_contour=False,
                 openpose_format='coco19',
                 **kwargs):
        super(OpenPose, self).__init__()

        self.use_hands = use_hands
        self.use_face = use_face
        self.model_type = model_type
        self.dtype = dtype
        self.joints_to_ign = joints_to_ign
        self.use_face_contour = use_face_contour

        self.openpose_format = openpose_format

        self.num_joints = (self.NUM_BODY_JOINTS +
                           2 * self.NUM_HAND_JOINTS * use_hands)

        self.keyp_folder = osp.join(data_folder, keyp_folder)

        self.keyp_paths = [osp.join(self.keyp_folder, keyp)
                    for keyp in os.listdir(self.keyp_folder)
                    if keyp.endswith('.json')]
        self.keyp_paths = sorted(self.keyp_paths)
        self.cnt = 0

    def get_model2data(self):
        return smpl_to_openpose(self.model_type, use_hands=self.use_hands,
                                use_face=self.use_face,
                                use_face_contour=self.use_face_contour,
                                openpose_format=self.openpose_format)

    def get_left_shoulder(self):
        return 2

    def get_right_shoulder(self):
        return 5

    def get_joint_weights(self):
        # The weights for the joint terms in the optimization
        optim_weights = np.ones(self.num_joints + 2 * self.use_hands +
                                self.use_face * 51 +
                                17 * self.use_face_contour,
                                dtype=np.float32)

        # Neck, Left and right hip
        # These joints are ignored because SMPL has no neck joint and the
        # annotation of the hips is ambiguous.
        if self.joints_to_ign is not None and -1 not in self.joints_to_ign:
            optim_weights[self.joints_to_ign] = 0.
        return torch.tensor(optim_weights, dtype=self.dtype)

    def __len__(self):
        return len(self.keyp_paths)

    def __getitem__(self, idx):
        keyp_path = self.keyp_paths[idx]
        return self.read_item(keyp_path)
########################################
### REMOVE RELIANCE ON IMG FOLDER ######
########################################
    def read_item(self, keyp_path):
        keyp_fn = osp.split(keyp_path)[1]
        keyp_fn, _ = osp.splitext(osp.split(keyp_path)[1])

        print(keyp_path, keyp_fn)
        keypoint_fn = osp.join(self.keyp_folder,
                               f'{keyp_fn}.json')
                               #img_fn + '_keypoints.json')
        
        print(keypoint_fn)
        
        keyp_tuple = read_keypoints(keypoint_fn, use_hands=self.use_hands,
                                    use_face=self.use_face,
                                    use_face_contour=self.use_face_contour)
       
        print(keyp_tuple.dims)

        if len(keyp_tuple.keypoints) < 1:
            return {}
        keypoints = np.stack(keyp_tuple.keypoints)

        output_dict = {'fn': keyp_fn,
                       'keyp_path': keyp_path,
                       'keypoints': keypoints,
                       'dims': keyp_tuple.dims}
        if keyp_tuple.gender_gt is not None:
            if len(keyp_tuple.gender_gt) > 0:
                output_dict['gender_gt'] = keyp_tuple.gender_gt
        if keyp_tuple.gender_pd is not None:
            if len(keyp_tuple.gender_pd) > 0:
                output_dict['gender_pd'] = keyp_tuple.gender_pd
        return output_dict

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.cnt >= len(self.keyp_paths):
            raise StopIteration

        keyp_path = self.keyp_paths[self.cnt]
        self.cnt += 1

        return self.read_item(keyp_path)
