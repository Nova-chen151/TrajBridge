from typing import Optional, Sequence

import torch


def normalize_tail_lens(tail_lens: Optional[Sequence[int]]) -> list:
    if tail_lens is None:
        return [0, 5, 10, 15]
    return [int(tail_len) for tail_len in tail_lens]


def select_tail_len(
    future_len: int,
    reconstruction_tail_lens: Optional[Sequence[int]],
    tail_len: Optional[int],
    mixed_tail_len: bool,
) -> int:
    tail_lens = normalize_tail_lens(reconstruction_tail_lens)
    if tail_len is not None:
        selected_tail_len = int(tail_len)
    elif mixed_tail_len:
        tail_idx = torch.randint(len(tail_lens), (1,)).item()
        selected_tail_len = tail_lens[tail_idx]
    else:
        selected_tail_len = tail_lens[0]

    if selected_tail_len < 0 or selected_tail_len > future_len:
        raise ValueError(f'tail_len must be in [0, {future_len}], got {selected_tail_len}')
    return selected_tail_len


def add_intermittent_recovery_masks(
    data,
    history_len: int,
    future_len: int,
    reconstruction_tail_lens: Optional[Sequence[int]] = None,
    tail_len: Optional[int] = None,
    mixed_tail_len: bool = True,
    gap_len: Optional[int] = None,
    window_protocol: str = 'fixed_total',
):
    """Attach explicit masks for intermittent observation trajectory recovery.

    All masks are full-timeline masks. ``gt_visible_mask`` preserves dataset
    availability, while ``model_input_mask`` marks only observations that the
    model may consume: historical observations plus the optional future tail.
    """
    gt_visible_mask = data['agent']['visible_mask'].clone()
    num_agents, num_steps = gt_visible_mask.shape
    device = gt_visible_mask.device

    if window_protocol not in ('fixed_gap', 'fixed_total'):
        raise ValueError(f'Unknown window_protocol: {window_protocol}')
    if history_len < 0 or future_len < 0:
        raise ValueError('history_len and future_len must be non-negative')
    if history_len > num_steps:
        raise ValueError(f'history_len {history_len} exceeds available steps {num_steps}')

    selected_tail_len = select_tail_len(
        future_len=future_len,
        reconstruction_tail_lens=reconstruction_tail_lens,
        tail_len=tail_len,
        mixed_tail_len=mixed_tail_len,
    )

    history_mask = torch.zeros(num_agents, num_steps, dtype=torch.bool, device=device)
    recovery_mask = torch.zeros(num_agents, num_steps, dtype=torch.bool, device=device)
    future_tail_mask = torch.zeros(num_agents, num_steps, dtype=torch.bool, device=device)

    if window_protocol == 'fixed_gap':
        if gap_len is None:
            raise ValueError('gap_len is required for window_protocol=fixed_gap')
        gap_len = int(gap_len)
        if gap_len < 0:
            raise ValueError('gap_len must be non-negative')
        effective_future_len = gap_len + selected_tail_len
    else:
        if gap_len is not None:
            raise ValueError('gap_len is only valid for window_protocol=fixed_gap')
        effective_future_len = future_len

    history_end = history_len
    future_end = history_end + effective_future_len
    if future_end > num_steps:
        raise ValueError(
            f'window ({history_len}+{effective_future_len}) exceeds available steps {num_steps}'
        )
    effective_tail_len = selected_tail_len
    tail_start = future_end - effective_tail_len

    history_mask[:, :history_end] = True
    recovery_mask[:, history_end:tail_start] = True
    if effective_tail_len > 0:
        future_tail_mask[:, tail_start:future_end] = True

    model_input_mask = gt_visible_mask & (history_mask | future_tail_mask)
    loss_mask = gt_visible_mask & recovery_mask

    data['agent']['gt_visible_mask'] = gt_visible_mask
    data['agent']['model_input_mask'] = model_input_mask
    data['agent']['history_mask'] = history_mask
    data['agent']['recovery_mask'] = recovery_mask
    data['agent']['future_tail_mask'] = future_tail_mask
    data['agent']['loss_mask'] = loss_mask

    # Backward-compatible alias for older reconstruction code and scripts.
    data['agent']['missing_mask'] = recovery_mask

    data['tail_len'] = torch.tensor(effective_tail_len, dtype=torch.long, device=device)
    data['missing_len'] = torch.tensor(effective_future_len - effective_tail_len, dtype=torch.long, device=device)
    return data
