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

def predict_action(observation, policy, device, use_amp):
    observation = copy(observation)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        # Convert to pytorch format: channel first and float32 in [0,1] with batch dimension
        for name in observation:
            if "image" in name:
                observation[name] = observation[name].type(torch.float32) / 255
                observation[name] = observation[name].permute(2, 0, 1).contiguous()
            observation[name] = observation[name].unsqueeze(0)
            observation[name] = observation[name].to(device)

        # Compute the next action with the policy
        # based on the current observation
        action = policy.select_action(observation)

        # Remove batch dimensifrom dataclasses import asdict, dataclasson
        action = action.squeeze(0)

        # Move to cpu, if not already the case
        action = action.to("cpu")

    return action

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
    joint_angles_rad[3] -= np.pi/2
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
        {"theta": 0, "d": 0.0563, "a": 0, "alpha": np.pi/2},
        {"theta": 0, "d": 0, "a": 0.108347, "alpha": np.pi},
        {"theta": 0, "d": 0, "a": 0.090467, "alpha": 0},
        {"theta": -np.pi/2, "d": 0, "a": 0, "alpha": -np.pi/2},
        {"theta": 0, "d": 0.05815, "a": 0, "alpha": 0},
    ]

    d = [0.0563, 0.0, 0.0, 0.0, 0.05815]
    a = [0.0, 0.108347, 0.090467, 0.0, 0.0]
    alpha = [np.pi/2, np.pi, 0, -np.pi/2, 0]

    # Compute positions for the new DH parameters
    positions_new, frames_new = compute_positions(dh_params_new)

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

    # Labels and view settings
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.set_title("New DH Frame Visualization")
    ax.set_ybound(-0.01, 0.01)
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

    
    # Load pretrained policy
    policy = None if c_cfg.policy is None else make_policy(c_cfg.policy, c_cfg.device, ds_meta=dataset.meta)
    device = c_cfg.device
    if isinstance(device, str):
        device = get_safe_torch_device(device)
    use_amp = c_cfg.use_amp

    # Create empty dictionary of: Observation, real action, predicted action

    # Loop through dataset
    #   Get observation (frames, poses)
    #   Pass inputs through policy
    #   Get action
    #   Save observation, real action, predicted action in dictionary
    results = []
    grasp_frame_obs = None
    grasp_orientation_obs = None
    grasp_frame_ra = None
    grasp_orientation_ra = None   
    grasp_frame_pa = None
    grasp_orientation_pa = None
    for idx in range(dataset.num_frames):

        obs_dict = {}
        obs_dict["observation.state"] = observations[idx]['observation.state']

        frames = {}
        for cap, name in zip(caps,dataset.meta.video_keys):
            ret, frames[name] = cap.read()
            obs_dict[f"{name}"] = torch.from_numpy(frames[name])
            if not ret:
                break            

        if policy is not None:
            pred_action = predict_action(obs_dict, policy, device, use_amp)
        
        action = actions[idx]["action"]
        result = {"observation": obs_dict["observation.state"], "real_action": action, "predicted_action": pred_action}
        results.append(result)

        # Get the Frame when grasping happens
        if result["real_action"][5] < 30 and grasp_frame_ra is None:
            grasp_frame_ra = idx
            grasp_orientation_ra = result["real_action"][4]
        if result["predicted_action"][5] < 30 and grasp_frame_pa is None:
            grasp_frame_pa = idx
            grasp_orientation_pa = result["real_action"][4]
        if result["observation"][5] < 30 and grasp_frame_obs is None:
            grasp_frame_obs = idx
            grasp_orientation_obs = result["observation"][4]
        

    # plot results in a graph
    real_actions = torch.stack([result["real_action"] for result in results]).numpy()
    predicted_actions = torch.stack([result["predicted_action"] for result in results]).numpy()
    observations = torch.stack([result["observation"] for result in results]).numpy()


    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    fig3d = plt.figure(figsize=(8, 6))
    ax3d = fig3d.add_subplot(111, projection='3d')
    e_fig, e_axs = plt.subplots(3, 1, figsize=(10, 15))

    # Compute trajectory
    end_eff_obs_traj = [forward_kinematics(joint_angles,d,a,alpha) for joint_angles in observations]
    end_eff_ra_traj = [forward_kinematics(joint_angles,d,a,alpha) for joint_angles in real_actions]
    end_eff_pa_traj = [forward_kinematics(joint_angles,d,a,alpha) for joint_angles in predicted_actions]

    # Convert to NumPy array for easy analysis
    end_eff_obs_traj = np.array(end_eff_obs_traj)
    end_eff_ra_traj = np.array(end_eff_ra_traj)
    end_eff_pa_traj = np.array(end_eff_pa_traj)

    joint_angles = observations[0][-6:]
    print(forward_kinematics(joint_angles,d,a,alpha))

    # Plot 2D trajectories
    trajectories = [end_eff_obs_traj, end_eff_ra_traj, end_eff_pa_traj]
    labels = ["End-Effector Observed", "End-Effector Real Action", "End-Effector Predicted Action"]
    titles = ["XY Projection", "XZ Projection", "YZ Projection"]
    axis_labels = [["Y","X"], ["X","Z"], ["Y","Z"]]
    rows = [[1,0],[0,2],[1,2]]
    colors = ["blue", "orange", "green"]

    diff_x = trajectories[1][:, 0] - trajectories[2][:, 0]
    diff_y = trajectories[1][:, 1] - trajectories[2][:, 1]
    diff_z = trajectories[1][:, 2] - trajectories[2][:, 2]
    diffs = [diff_x, diff_y, diff_z]

    diff_grasp_x = trajectories[1][grasp_frame_ra, 0] - trajectories[2][grasp_frame_pa, 0]
    diff_grasp_y = trajectories[1][grasp_frame_ra, 1] - trajectories[2][grasp_frame_pa, 1]
    diff_grasp_z = trajectories[1][grasp_frame_ra, 2] - trajectories[2][grasp_frame_pa, 2]
    diffs_grasp = [diff_grasp_x, diff_grasp_y, diff_grasp_z]
    
    for i, row in enumerate(rows):
        for j, trajectory in enumerate(trajectories):
            axs[i].plot(trajectory[:, row[0]], trajectory[:, row[1]], label=labels[j], color=colors[j])
            axs[i].scatter(trajectory[0, row[0]], trajectory[0, row[1]], color=colors[j], marker='o')
            axs[i].scatter(trajectory[-1, row[0]], trajectory[-1, row[1]], color=colors[j], marker='^')
            axs[i].scatter(trajectory[grasp_frame_obs, row[0]], trajectory[grasp_frame_obs, row[1]], color=colors[j], marker='x',s=70, label="Grasping")

        norm = np.linalg.norm([diffs[row[0]], diffs[row[1]]], axis=0)
        
        e_axs[i].plot(norm, label='Absolute Error', color = 'tab:blue')
        e_axs[i].scatter(grasp_frame_pa, norm[grasp_frame_pa], marker='x',s=70, color=colors[2])
        e_axs[i].scatter(grasp_frame_ra, norm[grasp_frame_ra], marker='x',s=70, color=colors[1])

        mean_error = np.mean(norm)
        rmse = np.sqrt(np.mean(norm**2))
        #r_squared = compute_r2(trajectories[1][:, row[0]],trajectories[0][:, row[1]])
        grasp_diff = np.linalg.norm([diffs_grasp[row[0]], diffs_grasp[row[1]]])
        
        e_axs[i].set_title(f'{titles[i]} - Mean Error: {mean_error:.2f}, RMSE: {rmse:.2f}, Grasping Diff: {grasp_diff:.4f}')
        e_axs[i].set_ylabel('Abs Error [m]')
        e_axs[i].legend()
        e_axs[i].set_xlabel('Frames')

        axs[i].set_title("End effector Trajectory " + titles[i])
        axs[i].set_xlabel(axis_labels[i][0] +" Position")
        axs[i].set_ylabel(axis_labels[i][1] +" Position")

        # Plot 3D trajectory
        ax3d.plot(trajectories[i][:, 0], trajectories[i][:, 1], trajectories[i][:, 2], color=colors[i], label=labels[i])
        ax3d.scatter(trajectories[i][0, 0], trajectories[i][0, 1], trajectories[i][0, 2], color=colors[i], marker='o')
        ax3d.scatter(trajectories[i][-1, 0], trajectories[i][-1, 1], trajectories[i][-1, 2], color=colors[i], marker='^')

    dh_params_new = [
        {"theta": np.radians(observations[0,0]), "d": 0.0563, "a": 0, "alpha": np.pi/2},
        {"theta": np.radians(observations[0,1]), "d": 0, "a": 0.108347, "alpha": np.pi},
        {"theta": np.radians(observations[0,2]), "d": 0, "a": 0.090467, "alpha": 0},
        {"theta": np.radians(observations[0,3])-np.pi/2, "d": 0, "a": 0, "alpha": -np.pi/2},
        {"theta": np.radians(observations[0,4]), "d": 0.05815, "a": 0, "alpha": 0},
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
    ''' 
    dh_params_new = [
        {"theta": np.radians(observations[100,0]), "d": 0.0563, "a": 0, "alpha": np.pi/2},
        {"theta": np.radians(observations[100,1]), "d": 0, "a": 0.108347, "alpha": np.pi},
        {"theta": np.radians(observations[100,2]), "d": 0, "a": 0.090467, "alpha": 0},
        {"theta": np.radians(observations[100,3])-np.pi/2, "d": 0, "a": 0, "alpha": -np.pi/2},
        {"theta": np.radians(observations[100,4]), "d": 0.05815, "a": 0, "alpha": 0},
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
    '''
    
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



