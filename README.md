# PoseNDF Implementation

## About
We are implementing [Pose-NDF](https://virtualhumans.mpi-inf.mpg.de/posendf/) paper .The goal of the paper is to create a continuous model for plausible human poses based on neural distance fields. We aim to recreate such a field in order to denoise made up motion capture data into a coherent and plausible movement.

![](https://virtualhumans.mpi-inf.mpg.de/posendf/teaser.png)

## Data Creation 
We use [AMASS](https://amass.is.tue.mpg.de/) dataset. This dataset holds all valid poses (0-distance). A SMPLX pose consists of 21 rotations $\theta_{i\in{0,20}}$. A single pose $\theta=\left(\theta_0, ..., \theta_{20} \right) $

In order to create a non-zero poses we do the following:
1. Choose a valid pose at random
2. Add normal distributed noise to it with varying sigmas
3. Project the new pose onto the 4D sphere (as quaternions are on the unit sphere in 4d space)
4. Out of all the valid poses, choose the closest to the random pose in l2 distance.
5. For these 100 calculate actual weighted quaternion distance (not l2).
6. Average the lowest 5 distances we got. Set this as the distance of the random pose to the valid poses.

![](https://user-images.githubusercontent.com/76158808/212336083-8ab281a2-7415-4082-b6de-a243fb50eb72.png)

** We use weighted sum to give more weight to rotations that are closer to the root as these affect their children as well as themselves.

** Notice that SMPLX uses axis angles as rotation representation while we use quaternions. When createing the data we switch to quaternions and when rendering the pose we switch back to axis angle.

### Creating a PoseDataset
In order to create a PoseDataset you need give the following parameters. 

On first call all data is created and saved to a pickle file in the data dir so that in the following calls, no data will need to be recreated.

```
data_dir (Path): path to dir contating (can be in subfolders) npz files as found in AMASS dataset
process_data (bool): Whether to recalculate non-zero poses from scratch.
zero_distance_pose_percentage (float, optional): The percenteage of the data that is the 0-set. Defaults to 1.0.
noise_sigma (List[float], optional): When creating a random pose, what is the std of noise to add to an existing pose. 
    The larger the sigma, the further the made up pose can get. Defaults to 0.
k_tag_neighbors: When creating random poses, for each random pose look for the `k_tag_neighbors` closest poses (in eucledian distance).
k_neighbors: for all the k_tag_neighbors we calculate the actual distance and take the k_neighbors closest poses. For them we calculate the mean distance.
weighted_sum: Whether to use weighted sum when calculating the distance of a random pose from the valid poses. 
    The weighted sum gives more weight the closer the rotation is to the root.
```
## Model Architecture
We implement the hierarchical pose encoding network using structural MLP consists of 7 hidden layers for distance field
prediction. We use ReLU as activation for the hidden layers.

## Training Process
After a pickle file was created, edit the config.yml file in order to define an experimental setup.
```
data:
        # Training data details
model:
        # Define the model architecture, number of hidden layers and activation.
train:
        # How many epochs, batch size, device, optimizer and num of workers.
```

Once configuration file is set, run the following:
```
python trainer.py --config=config.yml
```

## Denoising
The denoising process has 2 main options:
1. Creating X random poses and "denoisng" them to the manifold. (`denoising_random_motion.py`)
2. Taking an existing pose sequence adding noise to it, then denoising it. (`denoising_real_motion.py`)

Both scripts can be found under `denoising` folder. All inputs to the scripts are in capital letters at the beggining of the files.
There are 2 options for the pose projection onto the learnt manifold:

a. Using SGD to project the poses.

b. As the NN learnt a distance function, the output of the network should be the step size we need to take. As quaternions lie on a sphere and not on all $\R^4$, after the gradient step we project them to the sphere. We might need a few steps of this to reach a good distance to the sphere..
