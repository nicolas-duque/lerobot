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


def main():

    dh_params_new = [
        {"theta": 0,        "d": 0.0563,  "a": 0,       "alpha": np.pi/2},
        {"theta": (np.pi/2)-0.136,   "d": 0,       "a": 0.10935, "alpha": np.pi},
        {"theta": np.pi/2 - 0.162 -0.136 +0.162,    "d": 0,       "a": 0.10051,  "alpha": 0},
        {"theta": -np.pi/2 +np.pi/2, "d": 0,       "a": 0,       "alpha": -np.pi/2},
        {"theta": 0,        "d": 0.05815, "a": 0,       "alpha": 0},
    ]


    # Compute positions for the new DH parameters
    positions_new, frames_new = compute_positions(dh_params_new)
    
    # Plot the new DH parameter visualization
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot links
    ax.plot(positions_new[:, 0], positions_new[:, 1], positions_new[:, 2], "ko-", label="Links", linewidth=3)

    # Plot coordinate frames
    for i, frame in enumerate(frames_new):
        origin = frame[:3, 3]
        x_axis = frame[:3, 0] * 0.02  # Scale for visibility
        z_axis = frame[:3, 2] * 0.02

        ax.quiver(*origin, *x_axis, color="r", label="X-axis" if i == 0 else "",linewidth=2)
        ax.quiver(*origin, *z_axis, color="b", label="Z-axis" if i == 0 else "",linewidth=2)


    # Labels and view settings
    ax.set_xlabel("X-axis", fontsize=14)
    ax.set_ylabel("Y-axis", fontsize=14)
    ax.set_zlabel("Z-axis", fontsize=14)
    ax.set_title("DH Parameters Visualization",fontsize=20)
    ax.set_ybound(-0.1, 0.1)

    ax.legend(loc="upper left", fontsize=14, bbox_to_anchor=(0.15, 0.80))
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.view_init(elev=30, azim=45)


    fig.tight_layout()

    plt.show()



if __name__ == "__main__":
    main()



