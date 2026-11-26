from typing import List, Tuple

import torch
import torch.nn as nn

from utils import init_weights


class ProgressiveBridgeUnit(nn.Module):
    """One residual correction stage for a subset of the recovery gap."""

    def __init__(
        self,
        hidden_dim: int,
        residual_scale: float = 2.0,
        dropout: float = 0.1,
    ) -> None:
        super(ProgressiveBridgeUnit, self).__init__()
        self.residual_scale = residual_scale
        self.residual_mlp = nn.Sequential(
            nn.Linear(hidden_dim + 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
        )
        self.apply(init_weights)

    def forward(
        self,
        posterior_context: torch.Tensor,
        time_feature: torch.Tensor,
        gamma: torch.Tensor,
        stage_progress: torch.Tensor,
        stage_mask: torch.Tensor,
    ) -> torch.Tensor:
        distance_to_tail = 1.0 - gamma
        residual_input = torch.cat(
            [posterior_context, time_feature, gamma, distance_to_tail, stage_progress],
            dim=-1,
        )
        residual = self.residual_scale * torch.tanh(self.residual_mlp(residual_input))
        return residual * stage_mask.float()


class ProgressiveBridgeDecoder(nn.Module):
    """Progressively correct missing-gap frames from tail, middle, and history sides."""

    _ALIASES = {
        'tail': 'tail-side',
        'tail_side': 'tail-side',
        'tail-side': 'tail-side',
        'middle': 'middle',
        'mid': 'middle',
        'history': 'history-side',
        'history_side': 'history-side',
        'history-side': 'history-side',
    }

    def __init__(
        self,
        hidden_dim: int,
        num_historical_steps: int,
        num_future_steps: int,
        num_modes: int,
        num_bridge_stages: int = 3,
        bridge_stage_order: str = 'tail-side,middle,history-side',
        residual_scale: float = 2.0,
        dropout: float = 0.1,
        use_latest_visible_history_anchor: bool = False,
    ) -> None:
        super(ProgressiveBridgeDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.num_modes = num_modes
        self.num_bridge_stages = max(1, min(int(num_bridge_stages), 3))
        self.use_latest_visible_history_anchor = use_latest_visible_history_anchor
        self.stage_order = self._parse_stage_order(bridge_stage_order)[:self.num_bridge_stages]
        self.stage_units = nn.ModuleList([
            ProgressiveBridgeUnit(
                hidden_dim=hidden_dim,
                residual_scale=residual_scale,
                dropout=dropout,
            )
            for _ in self.stage_order
        ])

    @classmethod
    def _parse_stage_order(cls, bridge_stage_order: str) -> List[str]:
        if bridge_stage_order is None:
            bridge_stage_order = 'tail-side,middle,history-side'
        tokens = str(bridge_stage_order).replace('>', ',').replace('|', ',').split(',')
        stage_order = []
        for token in tokens:
            key = token.strip().lower().replace(' ', '_')
            if not key:
                continue
            if key not in cls._ALIASES:
                raise ValueError(f'Unknown bridge stage "{token}".')
            stage_name = cls._ALIASES[key]
            if stage_name not in stage_order:
                stage_order.append(stage_name)
        for stage_name in ('tail-side', 'middle', 'history-side'):
            if stage_name not in stage_order:
                stage_order.append(stage_name)
        return stage_order

    def _future_recovery_mask(self, data, num_agents: int, device: torch.device) -> torch.Tensor:
        if 'recovery_mask' in data['agent']:
            recovery_mask = data['agent']['recovery_mask']
        elif 'missing_mask' in data['agent']:
            recovery_mask = data['agent']['missing_mask']
        else:
            recovery_mask = torch.zeros(
                num_agents,
                self.num_historical_steps + self.num_future_steps,
                dtype=torch.bool,
                device=device,
            )
            if 'tail_len' in data:
                tail_len = data['tail_len'].to(device).long()
                if tail_len.dim() == 0:
                    tail_len = tail_len.unsqueeze(0)
                if 'batch' in data['agent']:
                    tail_len = tail_len[data['agent']['batch'].to(device)]
                else:
                    tail_len = tail_len[0].expand(num_agents)
                missing_len = torch.clamp(self.num_future_steps - tail_len, min=0, max=self.num_future_steps)
                step_idx = torch.arange(self.num_future_steps, device=device).view(1, -1)
                recovery_mask[:, self.num_historical_steps:self.num_historical_steps + self.num_future_steps] = (
                    step_idx < missing_len.view(-1, 1)
                )

        future = recovery_mask[:, self.num_historical_steps:self.num_historical_steps + self.num_future_steps]
        if future.size(1) == self.num_future_steps:
            return future.to(device)
        pad = future.new_zeros((future.size(0), self.num_future_steps - future.size(1)))
        return torch.cat([future, pad], dim=1).to(device)

    def _build_stage_masks(
        self,
        recovery_future_mask: torch.Tensor,
        posterior_valid: torch.Tensor,
        traj_prior: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_agents = recovery_future_mask.size(0)
        device = recovery_future_mask.device
        gap_len = recovery_future_mask.sum(dim=-1).clamp(min=1)
        rank = recovery_future_mask.long().cumsum(dim=-1) - 1
        normalized_rank = (rank.float() + 0.5) / gap_len.view(num_agents, 1).float()

        base_stage_masks = {
            'history-side': recovery_future_mask & (normalized_rank <= (1.0 / 3.0)),
            'middle': recovery_future_mask & (normalized_rank > (1.0 / 3.0)) & (normalized_rank <= (2.0 / 3.0)),
            'tail-side': recovery_future_mask & (normalized_rank > (2.0 / 3.0)),
        }

        active_agent = posterior_valid.view(num_agents, 1)
        gamma_base = torch.where(
            recovery_future_mask,
            (rank.float() + 1.0) / gap_len.view(num_agents, 1).float(),
            torch.zeros_like(normalized_rank),
        )
        gamma = gamma_base.view(num_agents, 1, 1, self.num_future_steps, 1).expand(
            -1,
            self.num_historical_steps,
            self.num_modes,
            -1,
            -1,
        ).contiguous()

        stage_masks = []
        for stage_name in self.stage_order:
            mask = base_stage_masks[stage_name] & active_agent
            mask = mask.view(num_agents, 1, 1, self.num_future_steps, 1).expand_as(gamma)
            stage_masks.append(mask)
        if stage_masks:
            stage_mask = torch.stack(stage_masks, dim=0)
        else:
            stage_mask = traj_prior.new_zeros((0,) + tuple(gamma.shape), dtype=torch.bool)

        stage_progress = normalized_rank.view(num_agents, 1, 1, self.num_future_steps, 1).expand_as(gamma)
        return stage_mask, gamma, stage_progress

    def forward(self, traj_prior: torch.Tensor, posterior_context: torch.Tensor, posterior_valid: torch.Tensor, data):
        num_agents = traj_prior.size(0)
        device = traj_prior.device
        recovery_future_mask = self._future_recovery_mask(data, num_agents, device)
        stage_mask, gamma, stage_progress = self._build_stage_masks(
            recovery_future_mask=recovery_future_mask,
            posterior_valid=posterior_valid.to(device),
            traj_prior=traj_prior,
        )

        time_idx = torch.linspace(0.0, 1.0, self.num_future_steps, device=device)
        time_feature = time_idx.view(1, 1, 1, self.num_future_steps, 1).expand_as(gamma)
        context = posterior_context.unsqueeze(3).expand(-1, -1, -1, self.num_future_steps, -1)

        traj_current = traj_prior
        stage_residuals = []
        aggregate_residual = traj_prior.new_zeros(traj_prior.shape)
        for stage_idx, bridge_unit in enumerate(self.stage_units):
            residual = bridge_unit(
                posterior_context=context,
                time_feature=time_feature,
                gamma=gamma,
                stage_progress=stage_progress,
                stage_mask=stage_mask[stage_idx],
            )
            stage_residuals.append(residual)
            aggregate_residual = aggregate_residual + residual
            traj_current = traj_current + gamma * residual

        if stage_residuals:
            stage_residual = torch.stack(stage_residuals, dim=0)
        else:
            stage_residual = traj_prior.new_zeros((0,) + tuple(traj_prior.shape))

        bridge_info = {
            'residual': aggregate_residual,
            'posterior_residual': aggregate_residual,
            'stage_residual': stage_residual,
            'stage_mask': stage_mask,
            'stage_names': tuple(self.stage_order),
            'gamma': gamma,
            'posterior_valid': posterior_valid,
            'recovery_future_mask': recovery_future_mask,
        }
        return traj_current, bridge_info
