from __future__ import annotations

from typing import Dict

import torch


def _history_mask(data, num_historical_steps: int, history_mask_key: str) -> torch.Tensor:
    agent = data['agent']
    if history_mask_key in agent:
        mask = agent[history_mask_key]
    elif 'history_input_mask' in agent:
        mask = agent['history_input_mask']
    else:
        mask = agent['visible_mask'][:, :num_historical_steps]
    mask = mask[:, :num_historical_steps].bool()
    if 'history_missing_mask' in agent:
        mask = mask & ~agent['history_missing_mask'][:, :num_historical_steps].bool()
    return mask


def compute_latest_visible_history_state(
    data,
    num_historical_steps: int,
    history_mask_key: str = 'history_input_mask',
) -> Dict[str, torch.Tensor]:
    """Return the last truly visible history state without reading hidden GT."""
    position = data['agent']['position']
    heading = data['agent']['heading']
    history_len = min(int(num_historical_steps), position.size(1))
    device = position.device
    num_agents = position.size(0)

    visible = _history_mask(data, history_len, history_mask_key)
    steps = torch.arange(history_len, device=device).view(1, -1).expand(num_agents, -1)
    anchor_valid = visible.any(dim=-1)
    anchor_index = torch.where(visible, steps, torch.zeros_like(steps)).max(dim=-1).values.long()

    agent_idx = torch.arange(num_agents, device=device)
    fallback_index = torch.zeros(num_agents, dtype=torch.long, device=device)
    safe_anchor_index = torch.where(anchor_valid, anchor_index, fallback_index)
    anchor_pos = position[agent_idx, safe_anchor_index]
    anchor_heading = heading[agent_idx, safe_anchor_index]

    before_anchor = visible & (steps < anchor_index.view(-1, 1))
    prev_valid = before_anchor.any(dim=-1)
    prev_index = torch.where(before_anchor, steps, torch.zeros_like(steps)).max(dim=-1).values.long()
    safe_prev_index = torch.where(prev_valid, prev_index, safe_anchor_index)
    prev_pos = position[agent_idx, safe_prev_index]

    delta_t = (safe_anchor_index - safe_prev_index).clamp(min=1).to(position.dtype)
    anchor_velocity = (anchor_pos - prev_pos) / delta_t.unsqueeze(-1)
    anchor_velocity = torch.where(prev_valid.unsqueeze(-1), anchor_velocity, torch.zeros_like(anchor_velocity))
    delta_t = torch.where(prev_valid, delta_t, torch.zeros_like(delta_t))

    return {
        'anchor_pos': anchor_pos,
        'anchor_heading': anchor_heading,
        'anchor_index': safe_anchor_index,
        'anchor_valid': anchor_valid,
        'prev_pos': prev_pos,
        'prev_index': safe_prev_index,
        'prev_valid': prev_valid,
        'anchor_velocity': anchor_velocity,
        'delta_t': delta_t,
    }
