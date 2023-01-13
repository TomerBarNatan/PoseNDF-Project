import torch
import numpy as np
import trimesh
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import sys
from tqdm import tqdm
import cv2
import os
sys.path.append("/home/or/school/PoseNDF-Project")

try:
    from smplx.body_models import SMPLX
except ImportError:
    raise ValueError("`pip install smplx` and download the SMPLX model data from https://smpl-x.is.tue.mpg.de/downloads")

from body_visualizer.mesh.mesh_viewer import MeshViewer
from train_utils.models import PoseNDF
from denoising.denoise_utils import render_pose, optimize_quaternion_poses_toward_gradient

torch.manual_seed(123)
np.random


NOISE_SIGMA = 0.02
GT_MOTION_PATH = '/Users/orlichter/Documents/school/amass/data/HDM05/mm/HDM_mm_01-02_03_120_stageii.npz'
PATH_TO_SMPLX_MODELS = "/home/or/school/smplx/models/smplx"  # this is the local path to th SMPL-X models files downloaded from here https://smpl-x.is.tue.mpg.de/downloads
PATH_TO_SAVE_IMAGES = "/home/or/school/PoseNDF-Project/denoising/results/real_motion"
PATH_TO_POSENDF_CHECKPOINT = "/home/or/school/posendf_model/small_data_l2_lrelu_l2_1e-05"
GOOD_ENOUGH_LOSS = 1e-3


def main():
    os.makedirs(PATH_TO_SAVE_IMAGES + '/gt', exist_ok=True)
    os.makedirs(PATH_TO_SAVE_IMAGES + '/noised', exist_ok=True)
    os.makedirs(PATH_TO_SAVE_IMAGES + '/denoised', exist_ok=True)

    model_path = f'{PATH_TO_SMPLX_MODELS}/SMPLX_NEUTRAL.npz'
    smplx_model = SMPLX(model_path=model_path)
    gt_poses_axis_angle = torch.from_numpy(np.load(GT_MOTION_PATH)['pose_body'].astype(np.float32)).view(-1, 21, 3)
    gt_poses_quaternion = torch.stack([torch.from_numpy(R.from_rotvec(x_i).as_quat()) for x_i in torch.unbind(gt_poses_axis_angle, dim=0)], dim=0)
    for i, pose in enumerate(gt_poses_axis_angle):
        image = render_pose(pose, smplx_model)
        cv2.imwrite(f'{PATH_TO_SAVE_IMAGES}/gt/{str(i).zfill(6)}.png', image)
        
    noised_poses_quaternion = gt_poses_quaternion + torch.normal(0, NOISE_SIGMA, gt_poses_quaternion.shape)
    noised_poses_quaternion /= torch.norm(noised_poses_quaternion, dim=2)
    noised_rotvec_poses = torch.stack([torch.from_numpy(R.from_quat(x_i).as_rotvec()) for x_i in torch.unbind(noised_poses_quaternion, dim=0)], dim=0).type(torch.float32)
    for i, pose in enumerate(noised_rotvec_poses):
        image = render_pose(pose, smplx_model)
        cv2.imwrite(f'{PATH_TO_SAVE_IMAGES}/noised/{str(i).zfill(6)}.png', image)
    
    optimized_quaternion_poses = optimize_quaternion_poses_toward_gradient(noised_poses_quaternion, PoseNDF.from_checkpoint_dir(Path(PATH_TO_POSENDF_CHECKPOINT))).detach().cpu()
    optimized_rotvec_poses = torch.stack([torch.from_numpy(R.from_quat(x_i).as_rotvec()) for x_i in torch.unbind(optimized_quaternion_poses, dim=0)], dim=0).type(torch.float32)
    for i, pose in enumerate(optimized_rotvec_poses):
        image = render_pose(pose, smplx_model)
        cv2.imwrite(f'{PATH_TO_SAVE_IMAGES}/denoised/{str(i).zfill(6)}.png', image)
