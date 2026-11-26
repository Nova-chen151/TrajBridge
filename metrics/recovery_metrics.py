import torch


def _zero(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def _last_valid_indices(mask: torch.Tensor):
    valid = mask.any(dim=-1)
    steps = torch.arange(mask.size(-1), device=mask.device).unsqueeze(0).expand_as(mask)
    last_idx = torch.where(mask, steps, torch.zeros_like(steps)).max(dim=-1).values
    return last_idx.long(), valid


def _first_last_indices(mask: torch.Tensor):
    valid = mask.any(dim=-1)
    steps = torch.arange(mask.size(-1), device=mask.device).unsqueeze(0).expand_as(mask)
    first = torch.where(mask, steps, torch.full_like(steps, mask.size(-1))).min(dim=-1).values
    last = torch.where(mask, steps, torch.zeros_like(steps)).max(dim=-1).values
    return first.long(), last.long(), valid


def gap_ADE(pred: torch.Tensor, target: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    errors = torch.norm(pred - target, p=2, dim=-1)
    if loss_mask.any():
        return errors[loss_mask].mean()
    return _zero(pred)


def gap_FDE(pred: torch.Tensor, target: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    errors = torch.norm(pred - target, p=2, dim=-1)
    last_idx, valid = _last_valid_indices(loss_mask)
    if valid.any():
        agent_idx = torch.arange(pred.size(0), device=pred.device)
        return errors[agent_idx[valid], last_idx[valid]].mean()
    return _zero(pred)


def boundary_error(pred: torch.Tensor, target: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    errors = torch.norm(pred - target, p=2, dim=-1)
    first_idx, last_idx, valid = _first_last_indices(loss_mask)
    if not valid.any():
        return _zero(pred)

    agent_idx = torch.arange(pred.size(0), device=pred.device)
    values = []
    safe_first_idx = first_idx.clamp(max=loss_mask.size(-1) - 1)
    safe_last_idx = last_idx.clamp(max=loss_mask.size(-1) - 1)

    first_valid = valid & loss_mask[agent_idx, safe_first_idx]
    if first_valid.any():
        values.append(errors[agent_idx[first_valid], safe_first_idx[first_valid]])
    last_valid = valid & loss_mask[agent_idx, safe_last_idx]
    if last_valid.any():
        values.append(errors[agent_idx[last_valid], safe_last_idx[last_valid]])
    if values:
        return torch.cat(values, dim=0).mean()
    return _zero(pred)


def _flatten_values(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if values.size(-1) > 1:
        values = torch.norm(values, p=2, dim=-1)
    else:
        values = values.squeeze(-1)
    if mask is not None:
        values = values[mask]
    return values.reshape(-1)


def _js_divergence_histogram(
    pred_values: torch.Tensor,
    target_values: torch.Tensor,
    pred_mask: torch.Tensor,
    target_mask: torch.Tensor,
    bins: int = 50,
    eps: float = 1e-8,
) -> torch.Tensor:
    pred_flat = _flatten_values(pred_values, pred_mask)
    target_flat = _flatten_values(target_values, target_mask)
    if pred_flat.numel() == 0 or target_flat.numel() == 0:
        return _zero(pred_values)

    combined = torch.cat([pred_flat, target_flat], dim=0)
    min_value = combined.min()
    max_value = combined.max()
    if torch.isclose(min_value, max_value):
        return _zero(pred_values)

    pred_hist = torch.histc(pred_flat, bins=bins, min=float(min_value.item()), max=float(max_value.item()))
    target_hist = torch.histc(target_flat, bins=bins, min=float(min_value.item()), max=float(max_value.item()))
    pred_prob = pred_hist / pred_hist.sum().clamp(min=eps)
    target_prob = target_hist / target_hist.sum().clamp(min=eps)
    mixture = 0.5 * (pred_prob + target_prob)
    js = 0.5 * (
        (pred_prob * ((pred_prob + eps) / (mixture + eps)).log()).sum()
        + (target_prob * ((target_prob + eps) / (mixture + eps)).log()).sum()
    )
    return js.to(pred_values.device)


def velocity_JS(pred: torch.Tensor, target: torch.Tensor, loss_mask: torch.Tensor, bins: int = 50) -> torch.Tensor:
    if pred.size(1) < 2:
        return _zero(pred)
    pred_velocity = pred[:, 1:] - pred[:, :-1]
    target_velocity = target[:, 1:] - target[:, :-1]
    velocity_mask = loss_mask[:, 1:] & loss_mask[:, :-1]
    return _js_divergence_histogram(pred_velocity, target_velocity, velocity_mask, velocity_mask, bins=bins)


def acceleration_JS(pred: torch.Tensor, target: torch.Tensor, loss_mask: torch.Tensor, bins: int = 50) -> torch.Tensor:
    if pred.size(1) < 3:
        return _zero(pred)
    pred_acceleration = pred[:, 2:] - 2.0 * pred[:, 1:-1] + pred[:, :-2]
    target_acceleration = target[:, 2:] - 2.0 * target[:, 1:-1] + target[:, :-2]
    acceleration_mask = loss_mask[:, 2:] & loss_mask[:, 1:-1] & loss_mask[:, :-2]
    return _js_divergence_histogram(pred_acceleration, target_acceleration, acceleration_mask, acceleration_mask, bins=bins)


def relative_distance_JS(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor,
    agent_batch: torch.Tensor = None,
    bins: int = 50,
) -> torch.Tensor:
    if pred.size(0) < 2:
        return _zero(pred)
    if agent_batch is None:
        agent_batch = pred.new_zeros(pred.size(0), dtype=torch.long)

    pred_distances = []
    target_distances = []
    distance_masks = []
    for batch_idx in agent_batch.unique(sorted=True):
        scene_mask = agent_batch == batch_idx
        if scene_mask.sum() < 2:
            continue
        scene_pred = pred[scene_mask]
        scene_target = target[scene_mask]
        scene_loss_mask = loss_mask[scene_mask]
        pair_i, pair_j = torch.triu_indices(scene_pred.size(0), scene_pred.size(0), offset=1, device=pred.device)
        pred_distance = torch.norm(scene_pred[pair_i] - scene_pred[pair_j], p=2, dim=-1)
        target_distance = torch.norm(scene_target[pair_i] - scene_target[pair_j], p=2, dim=-1)
        pair_mask = scene_loss_mask[pair_i] & scene_loss_mask[pair_j]
        pred_distances.append(pred_distance)
        target_distances.append(target_distance)
        distance_masks.append(pair_mask)

    if not pred_distances:
        return _zero(pred)

    pred_values = torch.cat(pred_distances, dim=0).unsqueeze(-1)
    target_values = torch.cat(target_distances, dim=0).unsqueeze(-1)
    mask = torch.cat(distance_masks, dim=0)
    return _js_divergence_histogram(pred_values, target_values, mask, mask, bins=bins)
