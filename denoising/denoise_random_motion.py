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
NUMBER_OF_POSES = 10
PATH_TO_SMPLX_MODELS = "/home/or/school/smplx/models/smplx"  # this is the local path to th SMPL-X models files downloaded from here https://smpl-x.is.tue.mpg.de/downloads
PATH_TO_SAVE_IMAGES = "/home/or/school/PoseNDF-Project/denoising/images"
PATH_TO_POSENDF_CHECKPOINT = "/home/or/school/posendf_model/small_data_l2_lrelu_l2_1e-05"
GOOD_ENOUGH_LOSS = 1e-3


def main():
    os.makedirs(PATH_TO_SAVE_IMAGES, exist_ok=True)

    model_path = f'{PATH_TO_SMPLX_MODELS}/SMPLX_NEUTRAL.npz'
    smplx_model = SMPLX(model_path=model_path)
    random_poses_axis_angles = torch.normal(0, NOISE_SIGMA, (NUMBER_OF_POSES, 21, 3), dtype=torch.float32)
    random_poses_axis_angles[0] = torch.tensor((0, 0, 0))
    random_poses_cumulated_axis_angles = torch.cumsum(random_poses_axis_angles, dim=0)
    random_poses_cumulated_quaternion = torch.stack([torch.from_numpy(R.from_rotvec(x_i).as_quat()) for x_i in torch.unbind(random_poses_cumulated_axis_angles, dim=0)], dim=0)
    for i, pose in enumerate(random_poses_cumulated_axis_angles):
        image = render_pose(pose, smplx_model)
        cv2.imwrite(f'{PATH_TO_SAVE_IMAGES}/before_optimization_{str(i).zfill(6)}.png', image)
    optimized_quaternion_poses = optimize_quaternion_poses_toward_gradient(random_poses_cumulated_quaternion, PoseNDF.from_checkpoint_dir(Path(PATH_TO_POSENDF_CHECKPOINT))).detach().cpu()
    optimized_rotvec_poses = torch.stack([torch.from_numpy(R.from_quat(x_i).as_rotvec()) for x_i in torch.unbind(optimized_quaternion_poses, dim=0)], dim=0).type(torch.float32)
    for i, pose in enumerate(optimized_rotvec_poses):
        image = render_pose(pose, smplx_model)
        cv2.imwrite(f'{PATH_TO_SAVE_IMAGES}/after_optimization_{str(i).zfill(6)}.png', image)


if __name__ == '__main__':
    main()