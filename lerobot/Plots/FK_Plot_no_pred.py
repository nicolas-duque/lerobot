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
    joint_angles_rad[5] *= -1 
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
        T = T@Ti # Multiply transformation matrices
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

    # Compute positions for the new DH parameters
    positions_new, frames_new = compute_positions(dh_params_new)
    #positions_mk, frames_mk = compute_positions(dh_params_mk)

    # Plot the new DH parameter visualization
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot links
    ax.plot(positions_new[:, 0], positions_new[:, 1], positions_new[:, 2], "bo-", label="Links")

    # Plot coordinate frames
    for i, frame in enumerate(frames_new):
        origin = frame[:3, 3]
        x_axis = frame[:3, 0] * 0.02  # Scale for visibility
        z_axis = frame[:3, 2] * 0.002

        ax.quiver(*origin, *x_axis, color="r", label="X-axis" if i == 0 else "")
        ax.quiver(*origin, *z_axis, color="b", label="Z-axis" if i == 0 else "")

    # Plot box frame
    R_dice = np.eye(3)
    P_dice = [0.19614778217500264,-0.07844556906171611,0.0]

    T_dice = np.eye(4)
    T_dice[:3,:3] = R_dice
    T_dice[:3,3] = P_dice

    print("T_dice: ", T_dice)

    origin = T_dice[:3, 3]
    x_axis = T_dice[:3, 0] * 0.02  # Scale for visibility
    y_axis = T_dice[:3, 1] * 0.02  # Scale for visibility
    z_axis = T_dice[:3, 2] * 0.002

    ax.quiver(*origin, *x_axis, color="r", label="X-axis" if i == 0 else "")
    ax.quiver(*origin, *y_axis, color="g", label="Y-axis" if i == 0 else "")
    ax.quiver(*origin, *z_axis, color="b", label="Z-axis" if i == 0 else "")

    # Labels and view settings
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.set_title("DH Frame Visualization")
    ax.set_ybound(-0.1, 0.1)
    #ax.set_xbound(-0.01, 0.015)
    ax.legend()
    ax.view_init(elev=30, azim=45)

    
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
    actions = dataset.hf_dataset.select_columns("action")
    joints = dataset.meta.info['features']['observation.state']['names']

    video_paths = [
         dataset.root / dataset.meta.get_video_file_path(c_cfg.episode,key) for key in dataset.meta.video_keys
    ]

    caps = [ cv2.VideoCapture(path) for path in video_paths]
    frames = [cap.get(cv2.CAP_PROP_FRAME_COUNT) for cap in caps]

    # Check if video opened successfully
    for cap in caps:
        if not cap.isOpened():
            print("Error: Could not open video.")
            exit()

    # Create empty dictionary of: Observation, real action, predicted action

    # Loop through dataset
    #   Get observation (frames, poses)
    #   Pass inputs through policy
    #   Get action
    #   Save observation, real action, predicted action in dictionary
    results = []
    grasp_frame_obs = None
    grasp_frame_ra = None
    for idx in range(dataset.num_frames):

        obs_dict = {}
        obs_dict["observation.state"] = observations[idx]['observation.state']

        frames = {}
        for cap, name in zip(caps,dataset.meta.video_keys):
            ret, frames[name] = cap.read()
            obs_dict[f"{name}"] = torch.from_numpy(frames[name])
            if not ret:
                break            
        
        action = actions[idx]["action"]
        result = {"observation": obs_dict["observation.state"], "real_action": action}
        results.append(result)

        # Get the Frame when grasping happens
        if result["real_action"][5] < 30 and grasp_frame_ra is None:
            grasp_frame_ra = idx
            grasp_orientation_ra = result["real_action"][4]
        if result["observation"][5] < 30 and grasp_frame_obs is None:
            grasp_frame_obs = idx
            grasp_orientation_obs = result["observation"][4]
        

    # plot results in a graph
    real_actions = torch.stack([result["real_action"] for result in results]).numpy()
    observations = torch.stack([result["observation"] for result in results]).numpy()


    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    fig3d = plt.figure(figsize=(8, 6))
    ax3d = fig3d.add_subplot(111, projection='3d')
    

    # Compute trajectory
    end_eff_obs_traj = [forward_kinematics(joint_angles,d,a,alpha) for joint_angles in observations]
    end_eff_ra_traj = [forward_kinematics(joint_angles,d,a,alpha) for joint_angles in real_actions]

    # Convert to NumPy array for easy analysis
    end_eff_obs_traj = np.array(end_eff_obs_traj)
    end_eff_ra_traj = np.array(end_eff_ra_traj)

    # Plot 2D trajectories
    trajectories = [end_eff_obs_traj, end_eff_ra_traj]
    labels = ["End-Effector Observed", "End-Effector Real Action"]
    titles = ["XY Projection", "XZ Projection", "YZ Projection"]
    axis_labels = [["Y","X"], ["X","Z"], ["Y","Z"]]
    rows = [[1,0],[0,2],[1,2]]
    colors = ["blue", "orange", "green"]
    
    for i, row in enumerate(rows):
        for j, trajectory in enumerate(trajectories):
            axs[i].plot(trajectory[:, row[0]], trajectory[:, row[1]], label=labels[j], color=colors[j])
            axs[i].scatter(trajectory[0, row[0]], trajectory[0, row[1]], color=colors[j], marker='o')
            axs[i].scatter(trajectory[-1, row[0]], trajectory[-1, row[1]], color=colors[j], marker='^')
            axs[i].scatter(trajectory[grasp_frame_obs, row[0]], trajectory[grasp_frame_obs, row[1]], color=colors[j], marker='x', label="Grasping")
    


        axs[i].set_title("End effector Trajectory " + titles[i])
        axs[i].set_xlabel(axis_labels[i][0] +" Position")
        axs[i].set_ylabel(axis_labels[i][1] +" Position")

        # Plot 3D trajectory
    ax3d.plot(trajectories[0][:, 0], trajectories[0][:, 1], trajectories[0][:, 2], color=colors[1], label=labels[0])
    ax3d.scatter(trajectories[0][0, 0], trajectories[0][0, 1], trajectories[0][0, 2], color=colors[1], marker='o')
    ax3d.scatter(trajectories[0][-1, 0], trajectories[0][-1, 1], trajectories[0][-1, 2], color=colors[1], marker='^')
    grasp = grasp_frame_obs
    dh_params_new = [
        {"theta": np.radians(observations[grasp,0]), "d": 0.0563, "a": 0, "alpha": np.pi/2},
        {"theta": np.radians(observations[grasp,1]), "d": 0, "a": 0.108347, "alpha": np.pi},
        {"theta": np.radians(observations[grasp,2]), "d": 0, "a": 0.090467, "alpha": 0},
        {"theta": np.radians(observations[grasp,3])-np.pi/2, "d": 0, "a": 0, "alpha": -np.pi/2},
        {"theta": np.radians(observations[grasp,4]), "d": 0.05815, "a": 0, "alpha": 0},
    ]
    dh_params_new = [
        {"theta": np.radians(observations[0,0]), "d": 0.0563, "a": 0, "alpha": np.pi/2},
        {"theta": np.radians(observations[0,1])-0.136, "d": 0, "a": 0.10935, "alpha": np.pi},
        {"theta": np.radians(observations[0,2]), "d": 0, "a": 0.10051, "alpha": 0},
        {"theta": np.radians(observations[0,3])-np.pi/2, "d": 0, "a": 0, "alpha": -np.pi/2},
        {"theta": -np.radians(observations[0,4]), "d": 0.05815, "a": 0, "alpha": 0},
    ]
    positions_new, frames_new = compute_positions(dh_params_new)

    print(frames_new)
    
    # Plot links
    ax3d.plot(positions_new[:, 0], positions_new[:, 1], positions_new[:, 2], "ko-", label="Old Links")

    # Plot coordinate frames
    for i, frame in enumerate(frames_new):
        origin = frame[:3, 3]
        x_axis = frame[:3, 0] * 0.02  # Scale for visibility
        z_axis = frame[:3, 2] * 0.02

        ax3d.quiver(*origin, *x_axis, color="r", label="X-axis" if i == 10 else "")
        ax3d.quiver(*origin, *z_axis, color="k", label="Z-axis" if i == 10 else "")
    
    
    dh_params_new = [
        {"theta": np.radians(observations[grasp,0]), "d": 0.0563, "a": 0, "alpha": np.pi/2},
        {"theta": np.radians(observations[grasp,1])-0.136, "d": 0, "a": 0.10935, "alpha": np.pi},
        {"theta": np.radians(observations[grasp,2])+0.162, "d": 0, "a": 0.10051, "alpha": 0},
        {"theta": np.radians(observations[grasp,3])-np.pi/2, "d": 0, "a": 0, "alpha": -np.pi/2},
        {"theta": -np.radians(observations[grasp,4]), "d": 0.05815, "a": 0, "alpha": 0},
    ]

    positions_new, frames_new = compute_positions(dh_params_new)
    # Plot links
    ax3d.plot(positions_new[:, 0], positions_new[:, 1], positions_new[:, 2], "bo-", label="Links")

    # Plot coordinate frames
    for i, frame in enumerate(frames_new):
        origin = frame[:3, 3]
        x_axis = frame[:3, 0] * 0.02  # Scale for visibility
        z_axis = frame[:3, 2] * 0.02

        ax3d.quiver(*origin, *x_axis, color="r", label="X-axis" if i == 0 else "")
        ax3d.quiver(*origin, *z_axis, color="b", label="Z-axis" if i == 0 else "")

    origin = T_dice[:3, 3]
    x_axis = T_dice[:3, 0] * 0.02  # Scale for visibility
    y_axis = T_dice[:3, 1] * 0.02  # Scale for visibility
    z_axis = T_dice[:3, 2] * 0.002

    x = T_dice@frame
    print(x)

    yaw = np.degrees(np.arctan2(T_dice[1,0],T_dice[0,0]))
    roll = np.degrees(np.arctan2(T_dice[2,1],T_dice[2,2]))
    pitch = np.degrees(np.arcsin(-T_dice[2,0]))

    yaw = np.degrees(np.arctan2(frame[1,0],frame[0,0]))
    roll = np.degrees(np.arctan2(frame[2,1],frame[2,2]))
    pitch = np.degrees(np.arcsin(-frame[2,0]))

    print(str(i)+": RPY: "+ str(roll) + ", " + str(pitch) + ", " + str(yaw))

    grasp_point = forward_kinematics(observations[grasp],d,a,alpha)

    ax3d.quiver(*origin, *x_axis, color="r", label="X-axis" if i == 0 else "")
    ax3d.quiver(*origin, *y_axis, color="g", label="Y-axis" if i == 0 else "")
    ax3d.quiver(*origin, *z_axis, color="b", label="Z-axis" if i == 0 else "")
    
    # Labels and legend
    ax3d.set_xlabel('X (m)')
    ax3d.set_ylabel('Y (m)')
    ax3d.set_zlabel('Z (m)')
    ax3d.set_title('End-Effector 3D Trajectory')
    ax3d.legend()
    ax3d.invert_yaxis()

    # Set viewing angle (optional)
    ax3d.view_init(elev=30, azim=45)

    fig.tight_layout()

    plt.show()



if __name__ == "__main__":
    main()



