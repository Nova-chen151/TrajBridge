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
    """Blend trajectories toward the observed tail with an optional social-force correction."""
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
            position_in_blend = t - blend_start
            normalized_pos = position_in_blend / blend_frames
            alpha = torch.sigmoid(torch.tensor((normalized_pos - 0.5) * 6, device=device, dtype=dtype))
            target_start = target_last_3[:, 0, :]

            current_pos = pred_traj[:, t, :]

            if use_social_force and t >= blend_start + 1:
                target_pos = target_start

                distance_to_target = torch.norm(target_pos - current_pos, dim=-1, keepdim=True)
                distance_threshold = torch.tensor(0.5, device=device, dtype=dtype)

                social_force_mask = (distance_to_target > distance_threshold).float()

                driving_force = (target_pos - current_pos) * force_strength * social_force_mask

                if t > 0:
                    pred_velocity = pred_traj[:, t, :] - pred_traj[:, t-1, :]
                else:
                    pred_velocity = torch.zeros_like(current_pos)

                remaining_steps = (F - 3) - t
                if remaining_steps > 0:
                    target_velocity = (target_last_3[:, -1, :] - target_start) / remaining_steps
                    velocity_alignment = (target_velocity - pred_velocity) * velocity_weight * social_force_mask
                else:
                    velocity_alignment = torch.zeros_like(current_pos)

                social_force = driving_force + velocity_alignment

                max_force = distance_to_target * 0.1
                social_force = torch.clamp(social_force, -max_force, max_force)

                adjusted_pred = current_pos + social_force
            else:
                adjusted_pred = current_pos

            smoothed[:, t, :] = (1 - alpha) * adjusted_pred + alpha * target_start

    return smoothed

def linear_smooth(pred_traj, target_last_3, transition_frames=5):
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

def exponential_smooth(pred_traj, target_last_3, transition_frames=5, exp_factor=2.0):
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
    """Smooth trajectories with a social-force particle filter and observed tail."""
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
        single_pred = pred_traj[n]  # [F, 2]
        single_target = target_last_3[n]  # [3, 2]
        
        if F > 1:
            v_initial = single_pred[1] - single_pred[0]
        else:
            v_initial = torch.zeros(2, device=device, dtype=dtype)
        
        target_direction = single_target[-1] - single_pred[blend_start]
        target_dist = torch.norm(target_direction)
        if target_dist > 1e-6:
            target_direction = target_direction / target_dist
        else:
            target_direction = torch.zeros(2, device=device, dtype=dtype)
            target_dist = torch.tensor(0.0, device=device, dtype=dtype)
        
        particles = torch.zeros(num_particles, 4, device=device, dtype=dtype)
        start_pos = single_pred[blend_start]
        particles[:, 0] = start_pos[0] + torch.randn(num_particles, device=device, dtype=dtype) * noise_pos
        particles[:, 1] = start_pos[1] + torch.randn(num_particles, device=device, dtype=dtype) * noise_pos
        particles[:, 2] = v_initial[0] + torch.randn(num_particles, device=device, dtype=dtype) * noise_vel
        particles[:, 3] = v_initial[1] + torch.randn(num_particles, device=device, dtype=dtype) * noise_vel
        
        dt = 0.1
        
        for t in range(blend_start, F):
            if t >= F - 3:
                target_idx = t - (F - 3)
                smoothed[n, t] = single_target[target_idx]
            else:
                for p in range(num_particles):
                    pos = particles[p, :2]
                    vel = particles[p, 2:]
                    
                    to_target = single_target[0] - pos
                    dist_to_target = torch.norm(to_target)
                    
                    dist_val = dist_to_target.item()
                    if dist_val > 1e-6:
                        desired_direction = to_target / dist_to_target
                    else:
                        desired_direction = torch.zeros(2, device=device, dtype=dtype)
                    
                    remaining_steps = (F - 3) - t
                    if remaining_steps > 0 and target_dist.item() > 1e-6:
                        desired_speed = target_dist / (remaining_steps * dt + 0.001)
                        desired_speed = torch.clamp(desired_speed, 0.0, 5.0)
                    else:
                        desired_speed = torch.tensor(0.0, device=device, dtype=dtype)
                    
                    desired_vel = desired_direction * desired_speed
                    force_driving = (desired_vel - vel) * 2.0
                    
                    to_pred = single_pred[t] - pos
                    force_pred = to_pred * 0.5
                    
                    total_force = force_driving + force_pred
                    
                    noise_v = torch.randn(2, device=device, dtype=dtype) * noise_vel
                    noise_p = torch.randn(2, device=device, dtype=dtype) * noise_pos
                    new_vel = vel + total_force * dt + noise_v
                    new_pos = pos + new_vel * dt + noise_p
                    
                    new_pos = torch.clamp(new_pos, -1000.0, 1000.0)
                    new_vel = torch.clamp(new_vel, -50.0, 50.0)
                    
                    particles[p, :2] = new_pos
                    particles[p, 2:] = new_vel
                
                pred_point = single_pred[t]
                distances = torch.norm(particles[:, :2] - pred_point.unsqueeze(0), dim=1)
                
                distances = torch.clamp(distances, 0.0, 10.0)
                weights = torch.exp(-distances ** 2 / (2 * noise_pos ** 2))
                
                weights_sum = weights.sum()
                if weights_sum.item() < 1e-10:
                    weights = torch.ones(num_particles, device=device, dtype=dtype) / num_particles
                else:
                    weights = weights / weights_sum
                
                smoothed_pos = (particles[:, :2] * weights.unsqueeze(1)).sum(dim=0)
                
                if t < F - 4:
                    weights_min = weights.min().item()
                    if weights_min < 1e-10:
                        weights_safe = weights + 1e-8
                    else:
                        weights_safe = weights
                    weights_safe = weights_safe / weights_safe.sum()
                    
                    # Sample on CPU to avoid CUDA multinomial assertions.
                    weights_cpu = weights_safe.cpu()
                    indices = torch.multinomial(weights_cpu, num_particles, replacement=True)
                    indices = indices.to(device)
                    particles = particles[indices]
                
                progress = (t - blend_start) / blend_frames if blend_frames > 0 else 1.0
                alpha = min(progress * 0.1, 0.1)
                
                target_point = single_target[0] + target_direction * target_dist * progress
                final_pos = (1 - alpha) * smoothed_pos + alpha * target_point
                
                smoothed[n, t] = final_pos
    
    return smoothed

def sigmoid_smooth(pred_traj, target_last_3, blend_frames=5):
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
    """Compute linear and angular velocity and acceleration from positions.

    Positions have shape [N, F, 2]; an optional [N, F] mask marks valid frames."""
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
    """Compute pairwise relative positions, velocities, and accelerations within a batch.

    Positions have shape [N, F, 2]; an optional [N, F] mask marks valid frames."""
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
    """Compute histogram-based Jensen-Shannon divergence over optional valid masks."""
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
