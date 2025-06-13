import logging
from dataclasses import asdict, dataclass
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pprint import pformat
from copy import copy
import torch
import cv2
from contextlib import nullcontext
import numpy as np
#import roboticstoolbox as rtb

# from safetensors.torch import load_file, save_file
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.factory import make_policy
from lerobot.common.robot_devices.control_configs import (
    ControlPipelineConfig,
    EvalControlConfig,
)
from lerobot.common.utils.utils import init_logging, get_safe_torch_device
from lerobot.configs import parser


# Function to compute transformation matrix using DH parameters
def dh_transform(theta, d, a, alpha):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0, sa, ca, d],
        [0, 0, 0, 1]
    ])

# Function to compute joint positions based on DH parameters
def compute_positions(dh_params):
    frames = [np.eye(4)]
    positions = [np.array([0, 0, 0])]  # Store joint positions
    
    for params in dh_params:
        T = dh_transform(params["theta"], params["d"], params["a"], params["alpha"])
        frames.append(frames[-1] @ T)
        positions.append(frames[-1][:3, 3])
    
    return np.array(positions), frames

# Forward kinematics function
def forward_kinematics(joint_angles,d,a,alpha):
    T = np.eye(4)
    #joint_angles[0] *= -1
    joint_angles_rad = np.radians(joint_angles)
    joint_angles_rad[1] -= 0.136
    joint_angles_rad[2] += 0.162
    joint_angles_rad[3] -= np.pi/2
    joint_angles_rad[4] *= -1 
    for i in range(5):
        theta = joint_angles_rad[i]
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha[i]), np.sin(alpha[i])
        Ti = np.array([
            [ct, -st * ca, st * sa, a[i] * ct],
            [st, ct * ca, -ct * sa, a[i] * st],
            [0, sa, ca, d[i]],
            [0, 0, 0, 1]
        ])
        print(T)
        T = T@Ti # Multiply transformation matrices
    print(T)
    return T[:3, 3]  # Extract position (x, y, z)

def compute_r2(q_d, q_a):
    mean_qd = np.mean(q_d)  # Mean of desired trajectory per joint
    sse = np.sum((q_d - q_a) ** 2)  # Sum of squared errors
    sst = np.sum((q_d - mean_qd) ** 2)  # Total variance
    r2 = 1 - (sse / sst)
    return r2


@parser.wrap()
def main(cfg: ControlPipelineConfig):

    dh_params_new = [
        {"theta": 0,        "d": 0.0563,  "a": 0,       "alpha": np.pi/2},
        {"theta": -0.136,   "d": 0,       "a": 0.10935, "alpha": np.pi},
        {"theta": +0.162,    "d": 0,       "a": 0.10051,  "alpha": 0},
        {"theta": -np.pi/2, "d": 0,       "a": 0,       "alpha": -np.pi/2},
        {"theta": 0,        "d": 0.05815, "a": 0,       "alpha": 0},
    ]

    d = [0.0563, 0.0, 0.0, 0.0, 0.05815]
    a = [0.0, 0.10935, 0.10051, 0.0, 0.0]
    alpha = [np.pi/2, np.pi, 0, -np.pi/2, 0]

    
    init_logging()
    logging.info(pformat(asdict(cfg)))

    # Load dataset
    #   Load videos
    #   Load observations
    #   Load actions

    # Create empty dataset or load existing saved episodes
    c_cfg: EvalControlConfig = cfg.control
    dataset = LeRobotDataset(c_cfg.repo_id, root=c_cfg.root, episodes=[c_cfg.episode])
    observations = dataset.hf_dataset.select_columns("observation.state")


    results = []
    grasp_frame_obs = None
    for idx in range(dataset.num_frames):

        obs_dict = {}
        obs_dict["observation.state"] = observations[idx]['observation.state']         
    
        result = {"observation": obs_dict["observation.state"]}
        results.append(result)

        if result["observation"][5] < 30 and grasp_frame_obs is None:
            grasp_frame_obs = idx
        

    # plot results in a graph
    observations = torch.stack([result["observation"] for result in results]).numpy()

    
    grasp = grasp_frame_obs    
    
    dh_params_new = [
        {"theta": np.radians(observations[grasp,0]), "d": 0.0563, "a": 0, "alpha": np.pi/2},
        {"theta": np.radians(observations[grasp,1])-0.136, "d": 0, "a": 0.10935, "alpha": np.pi},
        {"theta": np.radians(observations[grasp,2])+0.162, "d": 0, "a": 0.10051, "alpha": 0},
        {"theta": np.radians(observations[grasp,3])-np.pi/2, "d": 0, "a": 0, "alpha": -np.pi/2},
        {"theta": -np.radians(observations[grasp,4]), "d": 0.05815, "a": 0, "alpha": 0},
    ]

    positions_new, frames_new = compute_positions(dh_params_new)
    print(frames_new)
    # Plot links

    # Plot coordinate frames
    for i, frame in enumerate(frames_new):
        origin = frame[:3, 3]
        x_axis = frame[:3, 0] * 0.02  # Scale for visibility
        z_axis = frame[:3, 2] * 0.02


    yaw = np.degrees(np.arctan2(frame[1,0],frame[0,0]))
    roll = np.degrees(np.arctan2(frame[2,1],frame[2,2]))
    pitch = np.degrees(np.arcsin(-frame[2,0]))

    print(str(i)+": RPY: "+ str(roll) + ", " + str(pitch) + ", " + str(yaw))

    grasp_point = forward_kinematics(observations[grasp],d,a,alpha)



    plt.show()



if __name__ == "__main__":
    main()



