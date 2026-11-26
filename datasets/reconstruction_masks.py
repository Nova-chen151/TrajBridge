from typing import Optional, Sequence

import torch

from datasets.intermittent_masks import add_intermittent_recovery_masks
from datasets.intermittent_masks import select_tail_len


def build_target_agent_mask(
    data,
    target_agent_policy: str = 'category',
    target_categories: Sequence[int] = (1,),
) -> torch.Tensor:
    """Return the one target mask shared by loss, mode selection and metrics."""
    agent = data['agent']
    if target_agent_policy == 'category':
        if 'category' not in agent:
            raise KeyError('category is required for target_agent_policy=category')
        return torch.zeros_like(agent['category'], dtype=torch.bool).logical_or(
            torch.stack([agent['category'] == int(category) for category in target_categories]).any(dim=0)
        )
    if target_agent_policy == 'all_valid':
        mask = agent.get('gt_visible_mask', agent['visible_mask'])
        return mask.any(dim=-1)
    raise ValueError(f'Unknown target_agent_policy: {target_agent_policy}')


def add_reconstruction_masks(
    data,
    history_len: int,
    future_len: int,
    task_mode: str = 'reconstruction',
    reconstruction_tail_lens: Optional[Sequence[int]] = None,
    tail_len: Optional[int] = None,
    mixed_tail_len: bool = True,
    gap_len: Optional[int] = None,
    window_protocol: str = 'fixed_total',
):
    """Attach structural reconstruction masks to a HeteroData sample.

    Masks live on the full trajectory timeline. For the canonical 10+30 setup,
    this is frame 0..39, with loss restricted to the missing interval only.
    """
    visible_mask = data['agent']['visible_mask']
    num_agents, num_steps = visible_mask.shape
    device = visible_mask.device

    if task_mode in ('reconstruction', 'intermittent_recovery'):
        return add_intermittent_recovery_masks(
            data=data,
            history_len=history_len,
            future_len=future_len,
            reconstruction_tail_lens=reconstruction_tail_lens,
            tail_len=tail_len,
            mixed_tail_len=mixed_tail_len,
            gap_len=gap_len,
            window_protocol=window_protocol,
        )

    selected_tail_len = select_tail_len(
        future_len=future_len,
        reconstruction_tail_lens=reconstruction_tail_lens,
        tail_len=0,
        mixed_tail_len=False,
    )

    history_mask = torch.zeros(num_agents, num_steps, dtype=torch.bool, device=device)
    missing_mask = torch.zeros(num_agents, num_steps, dtype=torch.bool, device=device)
    future_tail_mask = torch.zeros(num_agents, num_steps, dtype=torch.bool, device=device)

    if history_len < 0 or future_len < 0:
        raise ValueError('history_len and future_len must be non-negative')
    if history_len + future_len > num_steps:
        raise ValueError(
            f'window ({history_len}+{future_len}) exceeds available steps {num_steps}'
        )
    history_end = history_len
    future_end = history_len + future_len
    effective_future_len = max(0, future_end - history_end)
    effective_tail_len = min(selected_tail_len, effective_future_len)
    tail_start = future_end - effective_tail_len

    history_mask[:, :history_end] = True
    if effective_tail_len > 0:
        future_tail_mask[:, tail_start:future_end] = True
    missing_mask[:, history_end:tail_start] = True

    model_input_mask = visible_mask & history_mask
    loss_mask = missing_mask & visible_mask

    data['agent']['gt_visible_mask'] = visible_mask.clone()
    data['agent']['model_input_mask'] = model_input_mask
    data['agent']['history_mask'] = history_mask
    data['agent']['recovery_mask'] = missing_mask
    data['agent']['missing_mask'] = missing_mask
    data['agent']['future_tail_mask'] = future_tail_mask
    data['agent']['loss_mask'] = loss_mask
    data['tail_len'] = torch.tensor(effective_tail_len, dtype=torch.long, device=device)
    data['missing_len'] = torch.tensor(effective_future_len - effective_tail_len, dtype=torch.long, device=device)
    return data
