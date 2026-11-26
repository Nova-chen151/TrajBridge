import torch
import math
import numpy as np

from typing import Any, List, Optional, Tuple, Union

# dt = 0.1s per frame (100ms)
DT = 0.1


def wrap_angle(angle: torch.Tensor, min_val: float = -math.pi, max_val: float = math.pi) -> torch.Tensor:
    return min_val + (angle + max_val) % (max_val - min_val)

def get_index_of_a_in_B(a: Optional[Any], list_B: Optional[List[Any]]) -> List[int]: 
    if not a or not list_B or a not in list_B:
        return []

    index = [list_B.index(a)]

    return index

def get_index_of_A_in_B(list_A: Optional[List[Any]], list_B: Optional[List[Any]]) -> List[int]: 
    if not list_A or not list_B:
        return []

    set_B = set(list_B)
    indices = [list_B.index(i) for i in list_A if i in set_B]

    return indices
    
def generate_clockwise_rotation_matrix(angle: torch.Tensor) -> torch.Tensor:
    matrix = torch.zeros_like(angle).unsqueeze(-1).repeat_interleave(2,-1).unsqueeze(-1).repeat_interleave(2,-1)
    matrix[..., 0, 0] = torch.cos(angle)
    matrix[..., 0, 1] = torch.sin(angle)
    matrix[..., 1, 0] = -torch.sin(angle)
    matrix[..., 1, 1] = torch.cos(angle)
    return matrix

def generate_counterclockwise_rotation_matrix(angle: torch.Tensor) -> torch.Tensor:
    matrix = torch.zeros_like(angle).unsqueeze(-1).repeat_interleave(2,-1).unsqueeze(-1).repeat_interleave(2,-1)
    matrix[..., 0, 0] = torch.cos(angle)
    matrix[..., 0, 1] = -torch.sin(angle)
    matrix[..., 1, 0] = torch.sin(angle)
    matrix[..., 1, 1] = torch.cos(angle)
    return matrix
    
def compute_angles_lengths_3D(vectors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    length = torch.norm(vectors, dim=-1)
    theta = torch.atan2(vectors[..., 1], vectors[..., 0])
    r_xy = torch.norm(vectors[..., :2], dim=-1)
    phi = torch.atan2(vectors[..., 2], r_xy)
    return length, theta, phi

def compute_angles_lengths_2D(vectors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    length = torch.norm(vectors, dim=-1)
    theta = torch.atan2(vectors[..., 1], vectors[..., 0])
    return length, theta

def drop_edge_between_samples(valid_mask: torch.Tensor, batch: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    if isinstance(batch, torch.Tensor):
        batch_matrix = batch.unsqueeze(-1) == batch.unsqueeze(-2)
    else:
        batch_src, batch_dst = batch
        batch_matrix = batch_src.unsqueeze(-1) == batch_dst.unsqueeze(-2)
    valid_mask = valid_mask * batch_matrix.unsqueeze(0)
    return valid_mask

def transform_traj_to_local_coordinate(traj: torch.Tensor, position: torch.Tensor, heading: torch.Tensor) -> torch.Tensor:
    traj = traj - position.unsqueeze(-2)
    rotation_matrix = generate_clockwise_rotation_matrix(heading)
    traj = torch.matmul(rotation_matrix.unsqueeze(-3), traj.unsqueeze(-1)).squeeze(-1)
    return traj

def transform_traj_to_global_coordinate(traj: torch.Tensor, position: torch.Tensor, heading: torch.Tensor) -> torch.Tensor:
    rotation_matrix = generate_counterclockwise_rotation_matrix(heading)
    traj = torch.matmul(rotation_matrix.unsqueeze(-3), traj.unsqueeze(-1)).squeeze(-1)
    traj = traj + position.unsqueeze(-2)
    return traj

def transform_point_to_local_coordinate(point: torch.Tensor, position: torch.Tensor, heading: torch.Tensor) -> torch.Tensor:
    point = point - position
    rotation_matrix = generate_clockwise_rotation_matrix(heading)
    point = torch.matmul(rotation_matrix, point.unsqueeze(-1)).squeeze(-1)
    return point

def transform_point_to_global_coordinate(point: torch.Tensor, position: torch.Tensor, heading: torch.Tensor) -> torch.Tensor:
    rotation_matrix = generate_counterclockwise_rotation_matrix(heading)
    point = torch.matmul(rotation_matrix, point.unsqueeze(-1)).squeeze(-1)
    point = point + position
    return point

def generate_target(position: torch.Tensor, mask: torch.Tensor, num_historical_steps: int, num_future_steps: int) -> Tuple[torch.Tensor, torch.Tensor]:
    target_traj = [position[:, i+1:i+1+num_future_steps] for i in range(num_historical_steps)]
    target_traj = torch.stack(target_traj, dim=1)
    target_mask = [mask[:,i+1:i+1+num_future_steps] for i in range(num_historical_steps)]
    target_mask = torch.stack(target_mask, dim=1)
    return target_traj, target_mask

def generate_reachable_matrix(edge_index: torch.Tensor, num_hops: int, max_nodes: int) -> list:
    values = torch.ones(edge_index.size(1), device=edge_index.device)
    sparse_mat = torch.sparse_coo_tensor(edge_index, values, torch.Size([max_nodes, max_nodes]))

    reach_matrices = []
    current_matrix = sparse_mat.clone()
    for _ in range(num_hops):
        current_matrix = current_matrix.coalesce()
        current_matrix = torch.sparse_coo_tensor(current_matrix.indices(), torch.ones_like(current_matrix.values()), current_matrix.size())

        edge_index_now = current_matrix.coalesce().indices()
        reach_matrices.append(edge_index_now)

        next_matrix = torch.sparse.mm(current_matrix, sparse_mat)

        current_matrix = next_matrix
    return reach_matrices

def generate_predict_mask(visible_mask: torch.Tensor, num_visible_steps: int) -> torch.Tensor:
    
    window = torch.ones((1, num_visible_steps), dtype=torch.float32, device=visible_mask.device)

    conv_result = torch.nn.functional.conv2d(visible_mask.float().unsqueeze(0).unsqueeze(0), window.unsqueeze(0).unsqueeze(0))
    conv_result = conv_result.squeeze(0).squeeze(0)
    
    predict_mask = conv_result == num_visible_steps
    predict_mask = torch.cat([torch.zeros((visible_mask.size(0), num_visible_steps-1), dtype=torch.bool, device=visible_mask.device), predict_mask], dim=1)
    
    return predict_mask

def sigmoid_smooth_social_force(pred_traj, target_last_3, blend_frames=5, force_strength=0.08, velocity_weight=0.05, use_social_force=True):
    """
    在sigmoid_smooth基础上加入社会力模型的思想 - 超保守版本

    Args:
        pred_traj: [N, F, 2] 预测轨迹
        target_last_3: [N, 3, 2] 真实轨迹最后3帧
        blend_frames: 混合帧数
        force_strength: 社会力强度系数 (极低以避免过度调整)
        velocity_weight: 速度对齐权重
        use_social_force: 是否启用社会力调整

    Returns:
        smoothed: [N, F, 2] 平滑后的轨迹
    """
    N, F, _ = pred_traj.shape
    device = pred_traj.device
    dtype = pred_traj.dtype
    smoothed = pred_traj.clone()

    if F <= 3:
        smoothed[:, -3:] = target_last_3
        return smoothed

    blend_frames = min(blend_frames, F - 3)
    blend_start = max(0, F - 3 - blend_frames)

    for t in range(blend_start, F):
        if t >= F - 3:
            target_idx = t - (F - 3)
            smoothed[:, t, :] = target_last_3[:, target_idx, :]
        else:
            # 基础的sigmoid混合
            position_in_blend = t - blend_start
            normalized_pos = position_in_blend / blend_frames
            alpha = torch.sigmoid(torch.tensor((normalized_pos - 0.5) * 6, device=device, dtype=dtype))
            target_start = target_last_3[:, 0, :]

            current_pos = pred_traj[:, t, :]

            if use_social_force and t >= blend_start + 1:  # 只在后半部分帧应用社会力
                # 计算社会力模型的驱动力 - 极低强度
                target_pos = target_start

                # 计算距离，只有距离较大时才应用社会力
                distance_to_target = torch.norm(target_pos - current_pos, dim=-1, keepdim=True)
                distance_threshold = torch.tensor(0.5, device=device, dtype=dtype)  # 距离阈值

                # 只有当距离超过阈值时才应用社会力
                social_force_mask = (distance_to_target > distance_threshold).float()

                # 简单的驱动力：朝向目标的微小调整
                driving_force = (target_pos - current_pos) * force_strength * social_force_mask

                # 计算预测轨迹的速度趋势
                if t > 0:
                    pred_velocity = pred_traj[:, t, :] - pred_traj[:, t-1, :]
                else:
                    pred_velocity = torch.zeros_like(current_pos)

                # 计算与目标轨迹的速度一致性 - 只在距离较大时应用
                remaining_steps = (F - 3) - t
                if remaining_steps > 0:
                    target_velocity = (target_last_3[:, -1, :] - target_start) / remaining_steps
                    velocity_alignment = (target_velocity - pred_velocity) * velocity_weight * social_force_mask
                else:
                    velocity_alignment = torch.zeros_like(current_pos)

                # 总的社会力
                social_force = driving_force + velocity_alignment

                # 限制社会力的最大幅度
                max_force = distance_to_target * 0.1  # 进一步降低最大调整幅度
                social_force = torch.clamp(social_force, -max_force, max_force)

                # 应用社会力调整预测轨迹
                adjusted_pred = current_pos + social_force
            else:
                # 不应用社会力，直接使用原始预测
                adjusted_pred = current_pos

            # 使用调整后的预测轨迹进行sigmoid混合
            smoothed[:, t, :] = (1 - alpha) * adjusted_pred + alpha * target_start

    return smoothed

def linear_smooth(pred_traj, target_last_3, transition_frames=5):  # 0.429907
    N, F, _ = pred_traj.shape
    smoothed = pred_traj.clone()
    
    if F <= 3:
        smoothed[:, -3:] = target_last_3
        return smoothed
    
    transition_start = max(0, F - 3 - transition_frames)
    
    for t in range(transition_start, F):
        if t >= F - 3:
            target_idx = t - (F - 3)
            smoothed[:, t, :] = target_last_3[:, target_idx, :]
        else:
            blend_ratio = (t - transition_start) / (F - 3 - transition_start)
            
            smoothed[:, t, :] = (1 - blend_ratio) * pred_traj[:, t, :] + blend_ratio * target_last_3[:, 0, :]
    
    return smoothed

def exponential_smooth(pred_traj, target_last_3, transition_frames=5, exp_factor=2.0):   # 0.451212
    N, F, _ = pred_traj.shape
    smoothed = pred_traj.clone()
    
    if F <= 3:
        smoothed[:, -3:] = target_last_3
        return smoothed
    
    device = pred_traj.device
    dtype = pred_traj.dtype
    
    transition_start = max(0, F - 3 - transition_frames)
    
    for t in range(transition_start, F):
        if t >= F - 3:
            target_idx = t - (F - 3)
            smoothed[:, t, :] = target_last_3[:, target_idx, :]
        else:
            normalized_pos = (t - transition_start) / (F - 3 - transition_start)
            weight_value = 1 - torch.exp(torch.tensor(-exp_factor * normalized_pos, device=device, dtype=dtype))
            weight = weight_value.item() if weight_value.dim() == 0 else weight_value
            smoothed[:, t, :] = (1 - weight) * pred_traj[:, t, :] + weight * target_last_3[:, 0, :]
    
    return smoothed


def social_force_smooth(pred_traj, target_last_3, num_particles=100, blend_frames=8, noise_pos=0.05, noise_vel=0.1):
    """
    基于社会力模型思想的粒子滤波平滑
    参考 SFPF_denoise 方法，使用物理模型引导 + 粒子滤波融合
    
    Args:
        pred_traj: [N, F, 2] 预测轨迹
        target_last_3: [N, 3, 2] 真实轨迹最后3帧
        num_particles: 粒子数量
        blend_frames: 混合帧数
        noise_pos: 位置噪声
        noise_vel: 速度噪声
    
    Returns:
        smoothed: [N, F, 2] 平滑后的轨迹
    """
    N, F, _ = pred_traj.shape
    device = pred_traj.device
    dtype = pred_traj.dtype
    
    smoothed = pred_traj.clone()
    
    if F <= 3:
        smoothed[:, -3:] = target_last_3
        return smoothed
    
    blend_frames = min(blend_frames, F - 3)
    blend_start = max(0, F - 3 - blend_frames)
    
    for n in range(N):
        # 获取单条轨迹
        single_pred = pred_traj[n]  # [F, 2]
        single_target = target_last_3[n]  # [3, 2]
        
        # 计算初始速度和方向（社会力模型中的驱动力）
        if F > 1:
            v_initial = single_pred[1] - single_pred[0]
        else:
            v_initial = torch.zeros(2, device=device, dtype=dtype)
        
        # 计算目标方向（从预测轨迹指向真实轨迹终点的期望方向）
        target_direction = single_target[-1] - single_pred[blend_start]
        target_dist = torch.norm(target_direction)
        if target_dist > 1e-6:
            target_direction = target_direction / target_dist
        else:
            target_direction = torch.zeros(2, device=device, dtype=dtype)
            target_dist = torch.tensor(0.0, device=device, dtype=dtype)
        
        # 初始化粒子：[position_x, position_y, velocity_x, velocity_y]
        particles = torch.zeros(num_particles, 4, device=device, dtype=dtype)
        start_pos = single_pred[blend_start]
        particles[:, 0] = start_pos[0] + torch.randn(num_particles, device=device, dtype=dtype) * noise_pos
        particles[:, 1] = start_pos[1] + torch.randn(num_particles, device=device, dtype=dtype) * noise_pos
        particles[:, 2] = v_initial[0] + torch.randn(num_particles, device=device, dtype=dtype) * noise_vel
        particles[:, 3] = v_initial[1] + torch.randn(num_particles, device=device, dtype=dtype) * noise_vel
        
        # 时间步长
        dt = 0.1
        
        for t in range(blend_start, F):
            if t >= F - 3:
                # 最后3帧直接使用真实值
                target_idx = t - (F - 3)
                smoothed[n, t] = single_target[target_idx]
            else:
                # 社会力模型更新粒子
                for p in range(num_particles):
                    pos = particles[p, :2]
                    vel = particles[p, 2:]
                    
                    # 驱动力：朝向目标的方向
                    to_target = single_target[0] - pos
                    dist_to_target = torch.norm(to_target)
                    
                    # 使用item()获取标量值进行比较，避免CUDA断言
                    dist_val = dist_to_target.item()
                    if dist_val > 1e-6:
                        desired_direction = to_target / dist_to_target
                    else:
                        desired_direction = torch.zeros(2, device=device, dtype=dtype)
                    
                    # 期望速度（基于剩余距离和时间）
                    remaining_steps = (F - 3) - t
                    if remaining_steps > 0 and target_dist.item() > 1e-6:
                        desired_speed = target_dist / (remaining_steps * dt + 0.001)
                        desired_speed = torch.clamp(desired_speed, 0.0, 5.0)
                    else:
                        desired_speed = torch.tensor(0.0, device=device, dtype=dtype)
                    
                    # 计算驱动力（期望速度与实际速度的差）
                    desired_vel = desired_direction * desired_speed
                    force_driving = (desired_vel - vel) * 2.0
                    
                    # 计算与预测轨迹的吸引力（引导粒子靠近预测轨迹）
                    to_pred = single_pred[t] - pos
                    force_pred = to_pred * 0.5
                    
                    # 总力
                    total_force = force_driving + force_pred
                    
                    # 更新速度和位置（考虑噪声）
                    noise_v = torch.randn(2, device=device, dtype=dtype) * noise_vel
                    noise_p = torch.randn(2, device=device, dtype=dtype) * noise_pos
                    new_vel = vel + total_force * dt + noise_v
                    new_pos = pos + new_vel * dt + noise_p
                    
                    # 数值稳定性检查
                    new_pos = torch.clamp(new_pos, -1000.0, 1000.0)
                    new_vel = torch.clamp(new_vel, -50.0, 50.0)
                    
                    particles[p, :2] = new_pos
                    particles[p, 2:] = new_vel
                
                # 计算权重（基于与预测轨迹的匹配度）
                pred_point = single_pred[t]
                distances = torch.norm(particles[:, :2] - pred_point.unsqueeze(0), dim=1)
                
                # 避免过大的距离导致数值问题
                distances = torch.clamp(distances, 0.0, 10.0)
                weights = torch.exp(-distances ** 2 / (2 * noise_pos ** 2))
                
                # 确保权重有效且不为全零
                weights_sum = weights.sum()
                if weights_sum.item() < 1e-10:
                    weights = torch.ones(num_particles, device=device, dtype=dtype) / num_particles
                else:
                    weights = weights / weights_sum
                
                # 加权平均作为平滑后的点（在重采样前计算）
                smoothed_pos = (particles[:, :2] * weights.unsqueeze(1)).sum(dim=0)
                
                # 重采样（仅在还有混合帧时进行）
                if t < F - 4:
                    # 确保权重严格正数
                    weights_min = weights.min().item()
                    if weights_min < 1e-10:
                        weights_safe = weights + 1e-8
                    else:
                        weights_safe = weights
                    weights_safe = weights_safe / weights_safe.sum()
                    
                    # 使用CPU进行multinomial避免CUDA断言问题
                    weights_cpu = weights_safe.cpu()
                    indices = torch.multinomial(weights_cpu, num_particles, replacement=True)
                    indices = indices.to(device)
                    particles = particles[indices]
                
                # 混合：随着接近真实轨迹，逐渐增加真实轨迹的权重
                progress = (t - blend_start) / blend_frames if blend_frames > 0 else 1.0
                alpha = min(progress * 0.1, 0.1)  # 最多30%的真实轨迹引导
                
                target_point = single_target[0] + target_direction * target_dist * progress
                final_pos = (1 - alpha) * smoothed_pos + alpha * target_point
                
                smoothed[n, t] = final_pos
    
    return smoothed

def sigmoid_smooth(pred_traj, target_last_3, blend_frames=5):   # 0.42717
    F = pred_traj.shape[1]
    smoothed = pred_traj.clone()
    
    if F <= 3:
        smoothed[:, -3:] = target_last_3
        return smoothed
    
    blend_frames = min(blend_frames, F - 3)
    blend_start = max(0, F - 3 - blend_frames)
    
    for t in range(blend_start, F):
        if t >= F - 3:
            target_idx = t - (F - 3)
            smoothed[:, t, :] = target_last_3[:, target_idx, :]
        else:
            position_in_blend = t - blend_start
            normalized_pos = position_in_blend / blend_frames
            alpha = torch.sigmoid(torch.tensor((normalized_pos - 0.5) * 6))
            target_start = target_last_3[:, 0, :]
            smoothed[:, t, :] = (1 - alpha) * pred_traj[:, t, :] + alpha * target_start
    
    return smoothed


def compute_kinematics_from_traj(pos: torch.Tensor, mask: Optional[torch.Tensor] = None, dt: float = DT) -> dict:
    """
    从轨迹计算速度、加速度、角速度、角加速度。
    pos: [N, F, 2], mask: [N, F] (optional, True=valid)
    Returns: dict with velocity [N,F-1,2], acceleration [N,F-2,2], angular_velocity [N,F-2], angular_acceleration [N,F-3]
    """
    N, F, _ = pos.shape
    device, dtype = pos.device, pos.dtype

    velocity = (pos[:, 1:] - pos[:, :-1]) / dt  # [N, F-1, 2]
    acceleration = (velocity[:, 1:] - velocity[:, :-1]) / dt  # [N, F-2, 2]

    # heading from position: atan2(dy, dx)
    dx = pos[:, 1:, 0] - pos[:, :-1, 0]
    dy = pos[:, 1:, 1] - pos[:, :-1, 1]
    heading = torch.atan2(dy, dx)  # [N, F-1]

    angular_velocity = wrap_angle(heading[:, 1:] - heading[:, :-1]) / dt  # [N, F-2]
    angular_acceleration = (angular_velocity[:, 1:] - angular_velocity[:, :-1]) / dt  # [N, F-3]

    out = {
        'velocity': velocity,
        'acceleration': acceleration,
        'angular_velocity': angular_velocity,
        'angular_acceleration': angular_acceleration,
    }
    if mask is not None:
        v_mask = mask[:, 1:] & mask[:, :-1]
        a_mask = v_mask[:, 1:] & v_mask[:, :-1]
        av_mask = v_mask[:, 1:] & v_mask[:, :-1]
        aa_mask = av_mask[:, 1:] & av_mask[:, :-1]
        out['velocity_mask'] = v_mask
        out['acceleration_mask'] = a_mask
        out['angular_velocity_mask'] = av_mask
        out['angular_acceleration_mask'] = aa_mask
    return out


def compute_relative_kinematics(pos: torch.Tensor, mask: Optional[torch.Tensor] = None, dt: float = DT) -> dict:
    """
    计算同一batch内智能体间的相对位置、相对速度、相对加速度。
    pos: [N, F, 2], N个智能体; mask: [N, F] True=有效
    Returns: dict with relative_position [P,F,2], relative_velocity [P,F-1,2], relative_acceleration [P,F-2,2]
    P = N*(N-1)//2
    """
    kin = compute_kinematics_from_traj(pos, mask, dt)
    N, F, _ = pos.shape

    rel_pos_list, rel_vel_list, rel_acc_list = [], [], []
    rel_pos_mask_list, rel_vel_mask_list, rel_acc_mask_list = [], [], []
    for i in range(N):
        for j in range(i + 1, N):
            rel_pos_list.append(pos[i] - pos[j])
            rel_vel_list.append(kin['velocity'][i] - kin['velocity'][j])
            rel_acc_list.append(kin['acceleration'][i] - kin['acceleration'][j])
            if mask is not None:
                rel_pos_mask_list.append(mask[i] & mask[j])
                v_mask = mask[i, 1:] & mask[i, :-1] & mask[j, 1:] & mask[j, :-1]
                rel_vel_mask_list.append(v_mask)
                a_mask = mask[i, 2:] & mask[i, 1:-1] & mask[i, :-2] & mask[j, 2:] & mask[j, 1:-1] & mask[j, :-2]
                rel_acc_mask_list.append(a_mask)

    out = {
        'relative_position': torch.stack(rel_pos_list, 0) if rel_pos_list else torch.zeros(0, F, 2, device=pos.device, dtype=pos.dtype),
        'relative_velocity': torch.stack(rel_vel_list, 0) if rel_vel_list else torch.zeros(0, F - 1, 2, device=pos.device, dtype=pos.dtype),
        'relative_acceleration': torch.stack(rel_acc_list, 0) if rel_acc_list else torch.zeros(0, F - 2, 2, device=pos.device, dtype=pos.dtype),
    }
    if mask is not None and rel_pos_list:
        out['relative_position_mask'] = torch.stack(rel_pos_mask_list, 0)
        out['relative_velocity_mask'] = torch.stack(rel_vel_mask_list, 0)
        out['relative_acceleration_mask'] = torch.stack(rel_acc_mask_list, 0)
    return out


def js_divergence_histogram(pred: torch.Tensor, target: torch.Tensor, pred_mask: Optional[torch.Tensor] = None,
                            target_mask: Optional[torch.Tensor] = None, num_bins: int = 50, eps: float = 1e-10) -> float:
    """
    基于直方图的JS散度。将连续值离散化后计算。
    pred, target: 任意形状，会被flatten。pred_mask, target_mask: 同形状，True=有效。
    Returns: JS divergence (scalar)
    """
    pred_np = pred.detach().cpu().numpy().flatten()
    target_np = target.detach().cpu().numpy().flatten()
    if pred_mask is not None:
        pred_np = pred_np[pred_mask.detach().cpu().numpy().flatten()]
    if target_mask is not None:
        target_np = target_np[target_mask.detach().cpu().numpy().flatten()]

    pred_np = pred_np[~np.isnan(pred_np) & np.isfinite(pred_np)]
    target_np = target_np[~np.isnan(target_np) & np.isfinite(target_np)]

    if len(pred_np) < 2 or len(target_np) < 2:
        return 0.0

    low = min(pred_np.min(), target_np.min())
    high = max(pred_np.max(), target_np.max())
    if high <= low:
        return 0.0
    bins = np.linspace(low, high, num_bins + 1)
    p_hist, _ = np.histogram(pred_np, bins=bins)
    q_hist, _ = np.histogram(target_np, bins=bins)
    p_hist = p_hist.astype(np.float64) + eps
    q_hist = q_hist.astype(np.float64) + eps
    P = p_hist / p_hist.sum()
    Q = q_hist / q_hist.sum()
    M = 0.5 * (P + Q) + eps
    kl_pm = np.sum(P * (np.log(P + eps) - np.log(M)))
    kl_qm = np.sum(Q * (np.log(Q + eps) - np.log(M)))
    js = 0.5 * kl_pm + 0.5 * kl_qm
    return float(js)