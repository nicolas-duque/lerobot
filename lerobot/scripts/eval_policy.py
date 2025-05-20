import logging
from dataclasses import asdict, dataclass
from pprint import pformat
from copy import copy
import torch
import cv2
from contextlib import nullcontext
import matplotlib.pyplot as plt
import numpy as np


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

        # Remove batch dimension
        action = action.squeeze(0)

        # Move to cpu, if not already the case
        action = action.to("cpu")

    return action


@parser.wrap()
def main(cfg: ControlPipelineConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    # Load dataset
    #   Load videos
    #   Load observations
    #   Load actions

    # Create empty dataset or load existing saved episodes
    c_cfg: EvalControlConfig = cfg.control
    #sanity_check_dataset_name(c_cfg.repo_id, c_cfg.policy)

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

    # plot results in a graph
    real_actions = torch.stack([result["real_action"] for result in results])
    predicted_actions = torch.stack([result["predicted_action"] for result in results])
    observations = torch.stack([result["observation"] for result in results])

    fig, axs = plt.subplots(6, 1, figsize=(10, 15))
    e_fig, e_axs = plt.subplots(6, 1, figsize=(10, 15))
    bp_fig = plt.figure(figsize=(10, 8))
    bp_ax = bp_fig.add_subplot(111)
    
    bp_ax.set_ylabel('Abs Error [°]')
    bp_ax.set_title('Absolute Error Boxplot')
    e_axs2 = []


    for i in range(6):
        obs = observations[:, i].numpy()
        re_act = real_actions[:, i].numpy()
        p_act = predicted_actions[:, i].numpy()
        axs[i].plot(obs, label='Observation')
        axs[i].plot(re_act, label='Real Action', linestyle='-.')
        axs[i].plot(p_act, label='Predicted Action', linestyle='--')
        axs[i].set_title(f'{joints[i]}')
        axs[i].set_ylabel('Angle [°]')
        axs[i].legend()

        e_axs2.append(e_axs[i].twinx())

        e_axs[i].plot(np.abs(re_act-p_act), label='Absolute Error', color = 'tab:blue')
        e_axs2[i].plot(np.abs(re_act-p_act)/np.abs(re_act), label='Relative Error', color = 'tab:red')

        mean_error = np.mean(np.abs(re_act-p_act))
        rmse = np.sqrt(np.mean((re_act-p_act)**2))
        r_squared = compute_r2(re_act, p_act)


        e_axs[i].set_title(f'{joints[i]} - Mean Error: {mean_error:.2f}, RMSE: {rmse:.2f}, R2: {r_squared:.2f}')

        e_axs[i].set_ylabel('Abs Error [°]',color = 'tab:blue')
        e_axs[i].legend()

        e_axs2[i].set_ylabel('Rel Error [%]',color = 'tab:red')
        e_axs2[i].legend()

        bp_ax.boxplot(np.abs(re_act-p_act), positions=[i], widths=0.6, patch_artist=True)

    bp_ax.set_xticks(np.arange(0,len(joints)),joints)
    fig.tight_layout()
    e_fig.tight_layout()
    plt.show()

def compute_r2(q_d, q_a):
    mean_qd = np.mean(q_d)  # Mean of desired trajectory per joint
    sse = np.sum((q_d - q_a) ** 2)  # Sum of squared errors
    sst = np.sum((q_d - mean_qd) ** 2)  # Total variance
    r2 = 1 - (sse / sst)
    return r2

if __name__ == "__main__":
    main()