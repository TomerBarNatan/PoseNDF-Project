
import torch
import numpy as np
import trimesh
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import sys
from tqdm import tqdm
import os
sys.path.append("/home/or/school/PoseNDF-Project")

from body_visualizer.mesh.mesh_viewer import MeshViewer


def render_pose(pose, smplx_model) -> np.ndarray:
    mesh_viewer = MeshViewer(width=512, height=512, use_offscreen=True)
    body_pose = torch.tensor(pose)[None]
    verts = smplx_model(body_pose=body_pose).vertices[0].cpu()
    mesh = trimesh.Trimesh(vertices=verts.cpu().detach().numpy(), faces=smplx_model.faces)
    mesh_viewer.set_static_meshes([mesh])
    return mesh_viewer.render(render_wireframe=False)


def optimize_quaternion_poses_sgd(quaternion_poses: torch.Tensor, model: torch.nn.Module, steps: int = 1000, lr: float=1e0,
                                  min_loss_threshold: float = 1e-3):
    """Use SGD to slowly move toward 0 distance pose

    Args:
        quaternion_poses (torch.Tensor): starting poses in quaternions
        model (torch.nn.Module): pose ndf model to evaluate the distace with
        steps (int, optional): Max number of steps to take. Defaults to 10000.
        lr (float, optional): SGD learning rate. Defaults to 1e0.
        min_loss_threshold (float, optional): When loss is below the threshold we can stop optimizing even if we haven't reached the max iterations.
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
        if loss < min_loss_threshold:
            print("Good enough loss achieved, stopping optimization")
            break
    return quaternion_poses


def optimize_quaternion_poses_toward_gradient(quaternion_poses: torch.Tensor, model: torch.nn.Module, steps: int = 1000,
                                              min_loss_threshold: float = 1e-3):
    """Optimize the quaternions moving toward the negative gradient. 
    The step size will be the distance function we evaluate as it theoretically should tell us what the distance to 0 is 
    (we still project to 4D sphere so it might take a few steps)

    Args:
        quaternion_poses (torch.Tensor): _description_
        model (torch.nn.Module): _description_
        steps (int, optional): _description_. Defaults to 1000.
        min_loss_threshold (float, optional): When loss is below the threshold we can stop optimizing even if we haven't reached the max iterations.

    Returns:
        torch.Tensor: optimized poses
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
        non_zero_distance_indices = dist_pred[:, 0] > 1e-8
        # Gradient step on non zero distances
        quaternion_poses.data[non_zero_distance_indices] = quaternion_poses[non_zero_distance_indices] - dist_pred[non_zero_distance_indices, None] * normalized_grad[non_zero_distance_indices]
        # Project to 4-d sphere
        quaternion_poses.data[non_zero_distance_indices] /= torch.norm(quaternion_poses.data[non_zero_distance_indices], dim=2, keepdim=True)
        tqdm_range.set_postfix_str(f'loss={loss.item(): .5f}')
        if loss < min_loss_threshold:
            print("Good enough loss achieved, stopping optimization")
            break
    return quaternion_poses