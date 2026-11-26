import torch
import torch.nn.functional as F


def _zero(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def _first_last_indices(mask: torch.Tensor):
    valid = mask.any(dim=-1)
    steps = torch.arange(mask.size(-1), device=mask.device).unsqueeze(0).expand_as(mask)
    first = torch.where(mask, steps, torch.full_like(steps, mask.size(-1))).min(dim=-1).values
    last = torch.where(mask, steps, torch.zeros_like(steps)).max(dim=-1).values
    return first.long(), last.long(), valid


def endpoint_consistency_loss(
    pred_future: torch.Tensor,
    target_future: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """Penalize first and last recovered endpoints against supervised targets."""
    if pred_future.size(1) == 0 or loss_mask.size(-1) == 0:
        return _zero(pred_future)

    first_idx, last_idx, valid = _first_last_indices(loss_mask)
    if not valid.any():
        return _zero(pred_future)

    agent_idx = torch.arange(pred_future.size(0), device=pred_future.device)
    losses = []
    safe_first_idx = first_idx.clamp(max=loss_mask.size(-1) - 1)
    safe_last_idx = last_idx.clamp(max=loss_mask.size(-1) - 1)

    first_valid = valid & loss_mask[agent_idx, safe_first_idx]
    if first_valid.any():
        losses.append(
            F.smooth_l1_loss(
                pred_future[agent_idx[first_valid], safe_first_idx[first_valid]],
                target_future[agent_idx[first_valid], safe_first_idx[first_valid]],
                reduction='none',
            ).sum(dim=-1).mean()
        )

    last_valid = valid & loss_mask[agent_idx, safe_last_idx]
    if last_valid.any():
        losses.append(
            F.smooth_l1_loss(
                pred_future[agent_idx[last_valid], safe_last_idx[last_valid]],
                target_future[agent_idx[last_valid], safe_last_idx[last_valid]],
                reduction='none',
            ).sum(dim=-1).mean()
        )

    if losses:
        return torch.stack(losses).mean()
    return _zero(pred_future)


def boundary_velocity_loss(
    pred_future: torch.Tensor,
    target_full: torch.Tensor,
    gt_visible_mask: torch.Tensor,
    history_len: int,
    recovery_mask: torch.Tensor,
    future_tail_mask: torch.Tensor,
) -> torch.Tensor:
    """Match velocities at history-gap and gap-tail boundaries."""
    if pred_future.size(1) == 0 or history_len <= 0:
        return _zero(pred_future)

    future_end = min(history_len + pred_future.size(1), target_full.size(1))
    recovery_future = recovery_mask[:, history_len:future_end]
    if recovery_future.size(1) < pred_future.size(1):
        pad = recovery_future.new_zeros((recovery_future.size(0), pred_future.size(1) - recovery_future.size(1)))
        recovery_future = torch.cat([recovery_future, pad], dim=-1)
    tail_future = future_tail_mask[:, history_len:future_end]
    if tail_future.size(1) < pred_future.size(1):
        pad = tail_future.new_zeros((tail_future.size(0), pred_future.size(1) - tail_future.size(1)))
        tail_future = torch.cat([tail_future, pad], dim=-1)

    first_recovery_idx, last_recovery_idx, has_recovery = _first_last_indices(recovery_future)
    first_tail_idx, _, has_tail = _first_last_indices(tail_future)
    agent_idx = torch.arange(pred_future.size(0), device=pred_future.device)
    losses = []

    hist_idx = min(history_len - 1, target_full.size(1) - 1)
    start_full_idx = (history_len + first_recovery_idx).clamp(max=target_full.size(1) - 1)
    start_valid = (
        has_recovery
        & gt_visible_mask[:, hist_idx]
        & gt_visible_mask[agent_idx, start_full_idx]
    )
    if start_valid.any():
        pred_start_vel = pred_future[agent_idx[start_valid], first_recovery_idx[start_valid]] - target_full[start_valid, hist_idx]
        target_start_vel = target_full[agent_idx[start_valid], start_full_idx[start_valid]] - target_full[start_valid, hist_idx]
        losses.append(F.smooth_l1_loss(pred_start_vel, target_start_vel, reduction='none').sum(dim=-1).mean())

    end_full_idx = (history_len + last_recovery_idx).clamp(max=target_full.size(1) - 1)
    tail_full_idx = (history_len + first_tail_idx).clamp(max=target_full.size(1) - 1)
    end_valid = (
        has_recovery
        & has_tail
        & gt_visible_mask[agent_idx, end_full_idx]
        & gt_visible_mask[agent_idx, tail_full_idx]
    )
    if end_valid.any():
        pred_end_vel = target_full[agent_idx[end_valid], tail_full_idx[end_valid]] - pred_future[agent_idx[end_valid], last_recovery_idx[end_valid]]
        target_end_vel = target_full[agent_idx[end_valid], tail_full_idx[end_valid]] - target_full[agent_idx[end_valid], end_full_idx[end_valid]]
        losses.append(F.smooth_l1_loss(pred_end_vel, target_end_vel, reduction='none').sum(dim=-1).mean())

    if losses:
        return torch.stack(losses).mean()
    return _zero(pred_future)


def jerk_smoothness_loss(
    full_traj: torch.Tensor,
    valid_mask: torch.Tensor,
    recovery_mask: torch.Tensor,
) -> torch.Tensor:
    """Smooth third-order finite differences on windows touching recovered frames."""
    if full_traj.size(1) < 4:
        return _zero(full_traj)

    jerk = full_traj[:, 3:] - 3.0 * full_traj[:, 2:-1] + 3.0 * full_traj[:, 1:-2] - full_traj[:, :-3]
    valid_window = valid_mask[:, 3:] & valid_mask[:, 2:-1] & valid_mask[:, 1:-2] & valid_mask[:, :-3]
    recovery_window = recovery_mask[:, 3:] | recovery_mask[:, 2:-1] | recovery_mask[:, 1:-2] | recovery_mask[:, :-3]
    mask = valid_window & recovery_window
    if mask.any():
        return jerk[mask].pow(2).sum(dim=-1).mean()
    return _zero(full_traj)
