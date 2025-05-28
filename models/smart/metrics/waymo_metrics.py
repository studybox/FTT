from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.protos import sim_agents_submission_pb2
from waymo_open_dataset.utils import trajectory_utils
from waymo_open_dataset.utils.sim_agents import submission_specs
from waymo_open_dataset.wdl_limited.sim_agents_metrics import metrics

import tensorflow as tf
import torch

def sample_simulated_states(simulated_states, num_roll_out=32, std_dev=0.5):
    """
    从单条 simulated_states 中生成多个 rollout，基于高斯采样。

    Args:
        simulated_states: [num_obj, 80, 4] torch.Tensor
        num_roll_out: 采样次数
        std_dev: 高斯采样的标准差（可设为 float 或 shape-compatible Tensor）

    Returns:
        [num_roll_out, num_obj, 80, 4] 的采样结果
    """
    num_obj, ts, dim = simulated_states.shape  # [num_obj, 80, 4]

    # 扩展为 [num_roll_out, num_obj, ts, dim]
    mean = simulated_states.unsqueeze(0).expand(num_roll_out, -1, -1, -1)  # [R, N, T, 4]

    # 构造高斯分布并采样
    noise = torch.randn_like(mean) * std_dev
    sampled_states = mean + noise

    return sampled_states

def joint_scene_from_states(states, object_ids):
    """Convert simulation states into a JointScene protobuf."""
    states = states.numpy()
    simulated_trajectories = [
        sim_agents_submission_pb2.SimulatedTrajectory(
            center_x=states[i, :, 0], center_y=states[i, :, 1],
            center_z=states[i, :, 2], heading=states[i, :, 3],
            object_id=object_ids[i]
        )
        for i in range(len(object_ids))
    ]
    return sim_agents_submission_pb2.JointScene(simulated_trajectories=simulated_trajectories)



def scenario_rollouts_from_states(scenario, states, object_ids):
    """Convert simulation states into a ScenarioRollouts protobuf."""
    joint_scenes = [joint_scene_from_states(states[i], object_ids) for i in range(states.shape[0])] # 32个，按照格式
    return sim_agents_submission_pb2.ScenarioRollouts(joint_scenes=joint_scenes, scenario_id=scenario.scenario_id)


def cal_waymo_metrics(scenario_proto, future_xy, num_roll_out=32):
    """
    scenario_proto: scenario in byte
    future_xy: [ROLL_OUT, num_obj, 80, 2]

    """

    # 1 提取log数据
    scenario_bytes = bytes(scenario_proto)
    scenario = scenario_pb2.Scenario.FromString(scenario_bytes)
    logged_trajectories = trajectory_utils.ObjectTrajectories.from_scenario(scenario) # 11 91（t）
    logged_trajectories = logged_trajectories.gather_objects_by_id(
        tf.convert_to_tensor(submission_specs.get_sim_agent_ids(scenario))
    ) # 8,91 （筛选出需要提交的）
    logged_trajectories = logged_trajectories.slice_time(
    start_index=0, end_index=submission_specs.CURRENT_TIME_INDEX + 1
        ) # 8,11过去时间
    
    # 2 处理预测数据
    ## 处理z logged_trajectories.z[n,his_ts] -> [n,ts,1] 最后一个维度扩展


    num_obj, future_ts = future_xy.shape[1], future_xy.shape[2]
    
    #TODO:处理z和heading
    #TODO:处理32rollout
    # (1) Extract last historical step's data
    logged_x = torch.tensor(logged_trajectories.x.numpy())           # [num_obj, 11]
    logged_y = torch.tensor(logged_trajectories.y.numpy())           # [num_obj, 11]
    logged_z = torch.tensor(logged_trajectories.z.numpy())           # [num_obj, 11]
    logged_heading = torch.tensor(logged_trajectories.heading.numpy())  # [num_obj, 11]
    last_x = logged_x[:, -1]      # [num_obj]
    last_y = logged_y[:, -1]
    last_z = logged_z[:, -1]
    last_heading = logged_heading[:, -1]
    
    # (2) Compute future headings (delta-based)
    # Concatenate last (x,y) with future (x,y) to compute heading changes
    # future_xy shape: [32, num_obj, 80, 2] (32 samples, N objects, 80 timesteps, 2 coordinates)
    # last_x, last_y, last_z shape: [num_obj]

    # (2) Compute future headings (delta-based)
    # Initialize shifted future_xy with same shape as future_xy
    future_xy_shifted = torch.roll(future_xy, shifts=1, dims=2)  # [32, num_obj, 80, 2]

    # For each sample (32), set the first timestep to use last historical (x,y)
    last_xy = torch.stack([last_x, last_y], dim=-1)  # [num_obj, 2]
    future_xy_shifted[:, :, 0, :] = last_xy.unsqueeze(0)  # Broadcast to [32, num_obj, 2]

    # Compute deltas (dx, dy)
    dx = future_xy[:, :, :, 0] - future_xy_shifted[:, :, :, 0]  # [32, num_obj, 80]
    dy = future_xy[:, :, :, 1] - future_xy_shifted[:, :, :, 1]  # [32, num_obj, 80]

    # Compute heading (arctan2(dy, dx))
    future_headings = torch.atan2(dy, dx)  # [32, num_obj, 80]

    # (3) Expand z-coordinate (constant for all future steps)
    future_z = last_z.unsqueeze(0).unsqueeze(2).expand(32, -1, future_ts)  # [32, num_obj, 80]

    # (4) Combine into simulated_states [32, num_obj, 80, 4]
    simulated_states = torch.stack([
        future_xy[:, :, :, 0],  # x [32, num_obj, 80]
        future_xy[:, :, :, 1],  # y [32, num_obj, 80]
        future_z,               # z [32, num_obj, 80]
        future_headings,        # heading [32, num_obj, 80]
    ], dim=-1)  # [32, num_obj, 80, 4]

    sampled_simulated_states = tf.convert_to_tensor(simulated_states.cpu().numpy(), dtype=tf.float32)

    # 3 计算指标
    object_ids = submission_specs.get_sim_agent_ids(scenario) # id，需要waymo scenario内置算法
    scenario_rollouts = scenario_rollouts_from_states(scenario, sampled_simulated_states, object_ids)

    # 校验 scenario_rollouts
    submission_specs.validate_scenario_rollouts(scenario_rollouts, scenario)

    # 计算指标
    config = metrics.load_metrics_config() 
    scenario_metrics = metrics.compute_scenario_metrics_for_bundle(config, scenario, scenario_rollouts)
    aggregate_metrics = metrics.aggregate_metrics_to_buckets(config, scenario_metrics)

    # 保存每个场景的指标
    scenario_metrics_dict = {
        "scenario_id": scenario.scenario_id,
        "scenario_metrics": scenario_metrics,
        "aggregate_metrics": aggregate_metrics
    }

    return scenario_metrics_dict