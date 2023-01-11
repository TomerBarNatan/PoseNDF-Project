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

torch.manual_seed(123)
np.random


NOISE_SIGMA = 0.02
NUMBER_OF_POSES = 10
PATH_TO_SMPLX_MODELS = "/home/or/school/smplx/models/smplx"  # this is the local path to th SMPL-X models files downloaded from here https://smpl-x.is.tue.mpg.de/downloads
PATH_TO_SAVE_IMAGES = "/home/or/school/PoseNDF-Project/denoising/images"
PATH_TO_POSENDF_CHECKPOINT = "/home/or/school/posendf_model/small_data_l2_lrelu_l2_1e-05"
GOOD_ENOUGH_LOSS = 1e-3


def render_pose(pose, smplx_model) -> np.ndarray:
    mesh_viewer = MeshViewer(width=512, height=512, use_offscreen=True)
    body_pose = torch.tensor(pose)[None]
    verts = smplx_model(body_pose=body_pose).vertices[0].cpu()
    mesh = trimesh.Trimesh(vertices=verts.cpu().detach().numpy(), faces=smplx_model.faces)
    mesh_viewer.set_static_meshes([mesh])
    return mesh_viewer.render(render_wireframe=False)


def optimize_quaternion_poses_sgd(quaternion_poses: torch.Tensor, model: torch.nn.Module, steps: int = 1000, lr: float=1e0):
    """Use SGD to slowly move toward 0 distance pose

    Args:
        quaternion_poses (torch.Tensor): starting poses in quaternions
        model (torch.nn.Module): pose ndf model to evaluate the distace with
        steps (int, optional): Max number of steps to take. Defaults to 10000.
        lr (float, optional): SGD learning rate. Defaults to 1e0.

    Returns:
        torch.Tensor: optimized poses
    """
    quaternion_poses = quaternion_poses.clone().type(torch.float32)
    quaternion_poses.requires_grad = True
    optimizer = torch.optim.SGD([quaternion_poses], lr=lr, momentum=0.9)
    loss = float('inf')
    tqdm_range = tqdm(range(steps))
    for _ in tqdm_range:
        optimizer.zero_grad()
        dist_pred = model(quaternion_poses)
        loss = dist_pred.mean()
        loss.backward()
        optimizer.step()
        quaternion_poses.data /= torch.norm(quaternion_poses.data, dim=2, keepdim=True)
        tqdm_range.set_postfix_str(f'loss={loss.item(): .5f}')
        if loss < GOOD_ENOUGH_LOSS:
            print("Good enough loss achieved, stopping optimization")
            break
    return quaternion_poses


def optimize_quaternion_poses_toward_gradient(quaternion_poses: torch.Tensor, model: torch.nn.Module, steps: int=1000):
    """Optimize the quaternions moving toward the negative gradient. 
    The step size will be the distance function we evaluate as it theoretically should tell us what the distance to 0 is 
    (we still project to 4D sphere so it might take a few steps)

    Args:
        quaternion_poses (torch.Tensor): _description_
        model (torch.nn.Module): _description_
        steps (int, optional): _description_. Defaults to 1000.

    Returns:
        _type_: _description_
    """
    quaternion_poses = quaternion_poses.clone().type(torch.float32)
    quaternion_poses.requires_grad = True
    loss = float('inf')
    tqdm_range = tqdm(range(steps))
    for _ in tqdm_range:
        dist_pred = model(quaternion_poses)
        loss = dist_pred.mean()
        loss.backward()
        normalized_grad = (quaternion_poses.grad.reshape(-1, 84) / torch.norm(quaternion_poses.grad.reshape(-1, 84), dim=1)[..., None]).reshape(-1, 21, 4)
        quaternion_poses.data = quaternion_poses - dist_pred[:, None] * normalized_grad
        quaternion_poses.data /= torch.norm(quaternion_poses.data, dim=2, keepdim=True)
        tqdm_range.set_postfix_str(f'loss={loss.item(): .5f}')
        if loss < GOOD_ENOUGH_LOSS:
            print("Good enough loss achieved, stopping optimization")
            break
    return quaternion_poses


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