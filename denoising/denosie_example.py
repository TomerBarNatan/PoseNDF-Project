import torch
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R


try:
    from smplx.body_models import SMPLX
except ImportError:
    raise ValueError("`pip install smplx` and download the SMPLX model data from https://smpl-x.is.tue.mpg.de/downloads")

from body_visualizer.mesh.mesh_viewer import MeshViewer
from train_utils.models import PoseNDF

torch.random.seed(123)
np.random


NOISE_SIGMA = 0.01
NUMBER_OF_POSES = 100
PATH_TO_SMPLX_MODELS = "/home/or/school/smplx/models/smplx"  # this is the local path to th SMPL-X models files downloaded from here https://smpl-x.is.tue.mpg.de/downloads


def render_pose(pose, smplx_model) -> np.ndarray:
    mesh_viewer = MeshViewer(width=512, height=512, use_offscreen=True)
    body_pose = torch.Tensor(pose).to(smplx_model.device)[None]
    verts = smplx_model(body_pose=body_pose).vertices[0].cpu()
    mesh = trimesh.Trimesh(vertices=verts.cpu().detach().numpy(), faces=smplx_model.faces)
    mesh_viewer.set_static_meshes([mesh])
    return mesh_viewer.render(render_wireframe=False)

def optimize_quaternion_poses(quaternion_poses, model):
    quaternion_poses = quaternion_poses.clone()
    quaternion_poses.req
    

def main():
    model_path = f'{PATH_TO_SMPLX_MODELS}/SMPLX_NEUTRAL.npz'
    smplx_model = SMPLX(model_path=model_path)
    random_poses_axis_angles = torch.normal(0, NOISE_SIGMA, (NUMBER_OF_POSES, 21, 3))
    random_poses_axis_angles[0] = torch.tensor((0, 0, 0))
    random_poses_cumulated_axis_angles = torch.cumsum(random_poses_axis_angles, dim=0)
    random_poses_cumulated_quaternion = torch.stack([torch.from_numpy(R.from_rotvec(x_i).as_quat()) for x_i in torch.unbind(random_poses_cumulated, dim=0)], dim=0)
    for pose in random_poses_cumulated_axis_angles:
        image = render_pose(pose, smplx_model)
    
    random_poses_cumulated_quaternion.requires_grad_ = True
    

if __name__ == '__main__':
    main()