import torch
from sklearn.neighbors import NearestNeighbors
import numpy as np
from torch.utils.data import Dataset
# from pytorch3d.transforms import axis_angle_to_quaternion
from scipy.spatial.transform import Rotation as R
from pathlib import Path
from tqdm import tqdm
import pickle as pkl

torch.manual_seed(123)


class PoseDataSet(Dataset):
    def __init__(self, data_dir: str, process_data: bool = True, zero_distance_pose_percentage: float = 0.99, noise_sigma: float = 0.01,
                 k_neighbors: int = 5, weighted_sum: bool = False, device='cpu'):
        """Read and load all available pose data from the data dir (AMASS Data set)
           The loaded poses are the 0-set poses and will have a 0 distance.
           
        Args:
            data_dir (Path): _description_
            process_data (bool): Whether to recalculate non-zero poses from scratch.
            zero_distance_pose_percentage (float, optional): The percenteage of the data that is the 0-set. Defaults to 1.0.
            noise_sigma (float, optional): When creating a random pose, what is the std of noise to add to an existing pose. 
                The larger the sigma, the further the made up pose can get. Defaults to 0.
            k_neighbors: When creating random poses, for each random pose look for the `k_neighbors` closest poses (in eucledian distance) 
                then for them calculate the mean quaternion distance.
            weighted_sum: Whether to use weighted sum when calculating the distance of a random pose from the valid poses. 
                The weighted sum gives more weight the closer the rotation is to the root.
        """
        self.data_dir = Path(data_dir)
        self.valid_poses = None
        self.device = device
        self.poses = []
        self.distances = []
        self.zero_distance_pose_percentage = zero_distance_pose_percentage
        self.k_neighbors = k_neighbors
        self.noise_sigma = noise_sigma
        self.pose_weights = torch.tensor([1] * 21) if not weighted_sum else torch.tensor(
            [7, 7, 7, 6, 6, 6, 5, 5, 5, 4, 4, 4, 4, 4, 3, 3, 3, 2, 2, 1, 1])
        self.pose_weights = torch.nn.functional.normalize(self.pose_weights.type(torch.float32), dim=0).to(self.device)
        
        if process_data or not (self.data_dir / 'processed_poses.pkl').exists():
            self._process_new_data()
        else:
            print(f"Loading preprocessed data from: `{self.data_dir / 'processed_poses.pkl'}`")
            with open(str(self.data_dir / 'processed_poses.pkl'), 'rb') as f:
                data = pkl.load(f)
                self.distances = data['distances']
                self.poses = data['poses']
        
    def _process_new_data(self):
        data_files = self.data_dir.rglob('*.npz')
        for mocap_file in data_files:
            mocap_data = np.load(mocap_file)
            if mocap_data.get('pose_body') is None:
                print(f"File {mocap_file} does not contain `pose_body` data.")
                continue
            pose_data_axis_angel = torch.from_numpy(np.load(mocap_file)['pose_body'].astype(np.float32)).view(-1, 21, 3).to(self.device)
            pose_data_quaternion = torch.stack([torch.from_numpy(R.from_rotvec(x_i).as_quat()) for x_i in torch.unbind(pose_data_axis_angel, dim=0)], dim=0)
            # pose_data_quaternion = axis_angle_to_quaternion(pose_data_axis_angel)
            self.poses.append(pose_data_quaternion.type(torch.float32))
        self.poses = torch.cat(self.poses).to(self.device)
        self.valid_poses = self.poses.clone()
        self.distances = torch.zeros(self.poses.shape[0], dtype=torch.float32)
        assert 0 < self.zero_distance_pose_percentage <= 1, f"`non_zero_pose_percentage` is {self.zero_distance_pose_percentage} and must be between 0-1"

        self.knn = NearestNeighbors(n_neighbors=self.k_neighbors)
        self.knn.fit(self.poses.cpu().numpy().reshape(-1, 84))
        
        # Create non zero poses
        amount_of_non_zero_poses = int(len(self.poses) / self.zero_distance_pose_percentage) - len(self.poses)
        pose_rotations = self._create_non_zero_pose(amount_of_non_zero_poses)
        distance = self._calculate_distance_to_zero_set(pose_rotations)
        self.poses = torch.cat([self.poses, pose_rotations])
        self.distances = torch.hstack([self.distances, distance])
        
        # double cover augmentation
        self._double_cover_augmentation()
        
        print(f"Saving processed data to: {self.data_dir / 'processed_poses.pkl'}")
        with open(str(self.data_dir / 'processed_poses.pkl'), 'wb') as f:
            pkl.dump({"poses": self.poses, "distances": self.distances}, f)
        
    def _create_non_zero_pose(self, amount_of_non_zero_poses) -> torch.Tensor:
        """Create a non zero pose in the following way:
        1. Choose a random valid pose
        2. Add to it normal noise with std of self.noise_sigma
        3. Normalize the poses as quaternions are of unit norm.

        Returns:
            torch.Tensor: a random non-zero pose. 21x4
        """
        idx = np.random.randint(0, self.valid_poses.shape[0], amount_of_non_zero_poses)  # TODO: add seed
        random_pose = self.valid_poses[idx].clone()
        random_pose += torch.normal(0, self.noise_sigma, random_pose.shape)
        random_pose /= np.linalg.norm(random_pose, axis=1, keepdims=True)
        return random_pose.type(torch.float32)

    def _calculate_distance_to_zero_set(self, poses_rotations) -> float:
        """ Calculate distance to zero set in the following way:
            1. Get the k nearest zero-set poses from the valid data
            2. calculate the mean quaternion distance from the pose_rotations to them.
            
            quaternion distance is calculated as follows:
            $\sqrt {(\sum_1^21 (w_i arccos(|<q_1, q_2>|)^2 ))}$
            

        Args:
            poses_rotations (_type_): pose of shape nx21x4

        Returns:
            float: _description_
        """
        distances = []
        for single_pose in tqdm(poses_rotations, desc="Calculating non-zero distance"):
            k_nearest_poses_indices = self.knn.kneighbors(single_pose.flatten()[None])[1]
            k_nearest_poses = self.valid_poses[k_nearest_poses_indices]
            quaternion_dists = torch.arccos(torch.abs(torch.einsum("bpq,pq->bp", k_nearest_poses, single_pose))) ** 2
            quaternion_dists = torch.sqrt(torch.sum(self.pose_weights * quaternion_dists, axis=1))
            distances.append(quaternion_dists.mean())
        return torch.tensor(distances, dtype=torch.float32).to(self.device)

    def _double_cover_augmentation(self):
        """
            Since a quaternion q and a quaternion -q are the same rotation, augment half of the quaternions to be -q
        """
        amount_of_quaternions = self.poses.shape[0] * self.poses.shape[1]
        idx = np.random.choice(range(amount_of_quaternions), amount_of_quaternions // 2, replace=False)
        self.poses.reshape(-1, 4)[idx] *= -1
                
    def __len__(self) -> int:
        return len(self.poses)

    def __getitem__(self, index):
        return self.poses[index], self.distances[index]


if __name__ == '__main__':
    amass_path = Path("/Users/orlichter/Documents/school/amass")
    dataset = PoseDataSet(amass_path)
    dataset[ - 10000]
    pass