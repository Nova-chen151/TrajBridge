from typing import Optional, Tuple

import torch
import torch.nn as nn

from datasets.history_anchor_utils import compute_latest_visible_history_state
from utils import init_weights


class FlowBridgeDecoder(nn.Module):
    """Endpoint-conditioned conditional flow matching for recovery gaps.

    Key shapes:
    - traj_prior: [N, H, K, F, 2]
    - posterior_context: [N, H, K, D]
    - target_traj: [N, F, 2], used only by flow_matching_loss during training.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_historical_steps: int,
        num_future_steps: int,
        num_modes: int,
        flow_sigma: float = 0.1,
        flow_train_sigma: Optional[float] = None,
        flow_inference_sigma: float = 0.0,
        flow_inference_init: str = 'deterministic',
        flow_ode_steps: int = 8,
        dropout: float = 0.1,
        use_latest_visible_history_anchor: bool = False,
    ) -> None:
        super(FlowBridgeDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.num_modes = num_modes
        self.flow_train_sigma = flow_sigma if flow_train_sigma is None else float(flow_train_sigma)
        self.flow_sigma = self.flow_train_sigma  # backward-compatible checkpoint/script alias
        self.flow_inference_sigma = float(flow_inference_sigma)
        if flow_inference_init not in ('deterministic', 'stochastic'):
            raise ValueError(f'Unknown flow_inference_init: {flow_inference_init}')
        self.flow_inference_init = flow_inference_init
        self.flow_ode_steps = max(1, int(flow_ode_steps))
        self.capture_flow_trace = False
        self.use_latest_visible_history_anchor = use_latest_visible_history_anchor

        # D + x_t(2) + t(1) + gamma(1) + stage one-hot(3)
        # + endpoint offsets(4) + tail_valid/recovery/tail flags(3)
        self.flow_mlp = nn.Sequential(
            nn.Linear(hidden_dim + 14, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, 2),
        )
        self.apply(init_weights)

    def _future_slice(self, tensor: torch.Tensor) -> torch.Tensor:
        future = tensor[:, self.num_historical_steps:self.num_historical_steps + self.num_future_steps]
        if future.size(1) == self.num_future_steps:
            return future
        pad_shape = list(future.shape)
        pad_shape[1] = self.num_future_steps - future.size(1)
        return torch.cat([future, future.new_zeros(pad_shape)], dim=1)

    def _future_mask(
        self,
        mask: Optional[torch.Tensor],
        num_agents: int,
        device: torch.device,
    ) -> torch.Tensor:
        if mask is None:
            return torch.zeros(num_agents, self.num_future_steps, dtype=torch.bool, device=device)
        mask = mask.to(device).bool()
        if mask.dim() != 2:
            raise ValueError(f'Expected mask with shape [N,T] or [N,F], got {tuple(mask.shape)}')
        if mask.size(1) == self.num_future_steps:
            return mask
        return self._future_slice(mask)

    def _masks_from_data(self, data, num_agents: int, device: torch.device):
        agent_data = data['agent']
        if 'recovery_mask' in agent_data:
            recovery_source = agent_data['recovery_mask']
        elif 'missing_mask' in agent_data:
            recovery_source = agent_data['missing_mask']
        else:
            recovery_source = None
        loss_source = agent_data['loss_mask'] if 'loss_mask' in agent_data else recovery_source
        tail_source = agent_data['future_tail_mask'] if 'future_tail_mask' in agent_data else None
        model_input_source = agent_data['model_input_mask'] if 'model_input_mask' in agent_data else agent_data['visible_mask']
        recovery_mask = self._future_mask(recovery_source, num_agents, device)
        loss_mask = self._future_mask(loss_source, num_agents, device)
        future_tail_mask = self._future_mask(tail_source, num_agents, device)
        model_input_mask = self._future_mask(model_input_source, num_agents, device)
        return recovery_mask, loss_mask, future_tail_mask, model_input_mask

    def _gap_features(self, recovery_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        num_agents = recovery_mask.size(0)
        gap_len = recovery_mask.sum(dim=-1).clamp(min=1)
        rank = recovery_mask.long().cumsum(dim=-1) - 1
        normalized_rank = torch.where(
            recovery_mask,
            (rank.float() + 0.5) / gap_len.view(num_agents, 1).float(),
            torch.zeros_like(rank, dtype=torch.float),
        )
        gamma = torch.where(
            recovery_mask,
            (rank.float() + 1.0) / gap_len.view(num_agents, 1).float(),
            torch.zeros_like(rank, dtype=torch.float),
        ).unsqueeze(-1)

        stage_index = torch.zeros_like(rank)
        stage_index = torch.where(normalized_rank > (1.0 / 3.0), torch.ones_like(stage_index), stage_index)
        stage_index = torch.where(normalized_rank > (2.0 / 3.0), torch.full_like(stage_index, 2), stage_index)
        stage_one_hot = torch.nn.functional.one_hot(stage_index.clamp(min=0), num_classes=3).float()
        stage_one_hot = stage_one_hot * recovery_mask.unsqueeze(-1).float()
        return gamma, stage_one_hot

    def _tail_endpoint(
        self,
        observed_future: torch.Tensor,
        future_tail_mask: torch.Tensor,
        model_input_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tail_observed_mask = future_tail_mask & model_input_mask
        valid = tail_observed_mask.any(dim=-1)
        steps = torch.arange(self.num_future_steps, device=observed_future.device).unsqueeze(0).expand_as(tail_observed_mask)
        first_idx = torch.where(tail_observed_mask, steps, torch.full_like(steps, self.num_future_steps)).min(dim=-1).values
        first_idx = first_idx.clamp(max=self.num_future_steps - 1).long()
        agent_idx = torch.arange(observed_future.size(0), device=observed_future.device)
        tail_endpoint = observed_future[agent_idx, first_idx]
        tail_endpoint = torch.where(valid.unsqueeze(-1), tail_endpoint, tail_endpoint.new_zeros(tail_endpoint.shape))
        return tail_endpoint, valid

    def _expand_agent_feature(self, feature: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        if x_t.dim() == 3:
            return feature
        if x_t.dim() == 5:
            return feature[:, None, None].expand(-1, x_t.size(1), x_t.size(2), -1, -1)
        raise ValueError(f'Expected x_t shape [N,F,2] or [N,H,K,F,2], got {tuple(x_t.shape)}')

    def _expand_context(self, posterior_context: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        if x_t.dim() == 3:
            if posterior_context.dim() == 4:
                posterior_context = posterior_context[:, -1, 0]
            return posterior_context[:, None].expand(-1, self.num_future_steps, -1)
        if x_t.dim() == 5:
            return posterior_context.unsqueeze(-2).expand(-1, -1, -1, self.num_future_steps, -1)
        raise ValueError(f'Expected x_t shape [N,F,2] or [N,H,K,F,2], got {tuple(x_t.shape)}')

    def _expand_time(self, t: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        t = torch.as_tensor(t, dtype=x_t.dtype, device=x_t.device)
        if t.dim() == 0:
            return x_t.new_full(x_t.shape[:-1] + (1,), float(t.item()))
        while t.dim() < len(x_t.shape[:-1]) + 1:
            t = t.unsqueeze(-1)
        return t.expand(x_t.shape[:-1] + (1,))

    def _velocity_field(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        posterior_context: torch.Tensor,
        recovery_mask: torch.Tensor,
        future_tail_mask: torch.Tensor,
        model_input_mask: torch.Tensor,
        history_endpoint: torch.Tensor,
        observed_future: torch.Tensor,
    ) -> torch.Tensor:
        gamma, stage_one_hot = self._gap_features(recovery_mask)
        tail_endpoint, tail_valid = self._tail_endpoint(
            observed_future=observed_future,
            future_tail_mask=future_tail_mask,
            model_input_mask=model_input_mask,
        )

        gamma = self._expand_agent_feature(gamma, x_t)
        stage_one_hot = self._expand_agent_feature(stage_one_hot, x_t)
        recovery_flag = self._expand_agent_feature(recovery_mask.unsqueeze(-1).float(), x_t)
        tail_flag = self._expand_agent_feature((future_tail_mask & model_input_mask).unsqueeze(-1).float(), x_t)
        context = self._expand_context(posterior_context, x_t)
        time_feature = self._expand_time(t, x_t)

        if x_t.dim() == 3:
            history = history_endpoint[:, None]
            tail = tail_endpoint[:, None]
            tail_valid_feature = tail_valid[:, None, None].float().expand(-1, self.num_future_steps, -1)
        else:
            history = history_endpoint[:, None, None, None].expand(-1, x_t.size(1), x_t.size(2), self.num_future_steps, -1)
            tail = tail_endpoint[:, None, None, None].expand(-1, x_t.size(1), x_t.size(2), self.num_future_steps, -1)
            tail_valid_feature = tail_valid[:, None, None, None, None].float().expand(-1, x_t.size(1), x_t.size(2), self.num_future_steps, -1)

        rel_history = x_t - history
        rel_tail = torch.where(tail_valid_feature.bool(), tail - x_t, torch.zeros_like(x_t))
        flow_input = torch.cat(
            [
                context,
                x_t,
                time_feature,
                gamma,
                stage_one_hot,
                rel_history,
                rel_tail,
                tail_valid_feature,
                recovery_flag,
                tail_flag,
            ],
            dim=-1,
        )
        return self.flow_mlp(flow_input)

    def _observations_from_data(self, data, num_agents: int, device: torch.device):
        position = data['agent']['position'].to(device)
        model_input_source = data['agent']['model_input_mask'] if 'model_input_mask' in data['agent'] else data['agent']['visible_mask']
        model_input_mask = self._future_mask(model_input_source, num_agents, device)
        observed_future = self._future_slice(position)
        if self.use_latest_visible_history_anchor:
            history_state = compute_latest_visible_history_state(data, self.num_historical_steps)
            history_endpoint = history_state['anchor_pos'].to(device)
        else:
            hist_idx = max(self.num_historical_steps - 1, 0)
            history_endpoint = position[:, hist_idx]
        return history_endpoint, observed_future, model_input_mask

    def _history_endpoint_from_tensors(
        self,
        history_traj: torch.Tensor,
        model_input_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if not self.use_latest_visible_history_anchor:
            return history_traj[:, max(history_traj.size(1) - 1, 0)]
        num_agents, history_len, _ = history_traj.shape
        device = history_traj.device
        if model_input_mask is None:
            history_visible = torch.ones(num_agents, history_len, dtype=torch.bool, device=device)
        else:
            model_input_mask = model_input_mask.to(device).bool()
            history_visible = model_input_mask[:, :history_len]
        steps = torch.arange(history_len, device=device).view(1, -1).expand(num_agents, -1)
        valid = history_visible.any(dim=-1)
        anchor_index = torch.where(history_visible, steps, torch.zeros_like(steps)).max(dim=-1).values.long()
        anchor_index = torch.where(valid, anchor_index, torch.zeros_like(anchor_index))
        return history_traj[torch.arange(num_agents, device=device), anchor_index]

    def forward(
        self,
        traj_prior: torch.Tensor,
        posterior_context: torch.Tensor,
        posterior_valid: torch.Tensor,
        data,
        target_traj: Optional[torch.Tensor] = None,
        recovery_mask: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        future_tail_mask: Optional[torch.Tensor] = None,
    ):
        del target_traj, loss_mask
        num_agents = traj_prior.size(0)
        device = traj_prior.device
        recovery_mask_data, _, future_tail_mask_data, model_input_mask = self._masks_from_data(data, num_agents, device)
        recovery_mask = self._future_mask(recovery_mask, num_agents, device) if recovery_mask is not None else recovery_mask_data
        future_tail_mask = self._future_mask(future_tail_mask, num_agents, device) if future_tail_mask is not None else future_tail_mask_data
        history_endpoint, observed_future, model_input_mask_data = self._observations_from_data(data, num_agents, device)
        model_input_mask = model_input_mask_data if model_input_mask_data is not None else model_input_mask

        if self.flow_inference_init == 'deterministic':
            x = traj_prior
        else:
            x = traj_prior + self.flow_inference_sigma * torch.randn_like(traj_prior)
        flow_trace = [x.detach()] if self.capture_flow_trace else None
        dt = 1.0 / float(self.flow_ode_steps)
        for step_idx in range(self.flow_ode_steps):
            t = traj_prior.new_tensor(step_idx * dt)
            velocity = self._velocity_field(
                x_t=x,
                t=t,
                posterior_context=posterior_context,
                recovery_mask=recovery_mask,
                future_tail_mask=future_tail_mask,
                model_input_mask=model_input_mask,
                history_endpoint=history_endpoint,
                observed_future=observed_future,
            )
            x = x + dt * velocity
            if flow_trace is not None:
                flow_trace.append(x.detach())

        recovery_flag = recovery_mask[:, None, None, :, None]
        tail_observed_flag = (future_tail_mask & model_input_mask)[:, None, None, :, None]
        observed_future_expanded = observed_future[:, None, None].expand_as(traj_prior)
        traj_final = torch.where(recovery_flag, x, traj_prior)
        traj_final = torch.where(tail_observed_flag, observed_future_expanded, traj_final)

        residual = (traj_final - traj_prior) * (recovery_flag | tail_observed_flag).float()
        gamma, stage_one_hot = self._gap_features(recovery_mask)
        stage_mask = stage_one_hot.permute(2, 0, 1).bool()
        stage_mask = stage_mask[:, :, None, None, :, None].expand(
            -1,
            -1,
            self.num_historical_steps,
            self.num_modes,
            -1,
            -1,
        )
        bridge_info = {
            'residual': residual,
            'posterior_residual': residual,
            'stage_mask': stage_mask,
            'gamma': gamma[:, None, None].expand(-1, self.num_historical_steps, self.num_modes, -1, -1),
            'posterior_valid': posterior_valid,
            'recovery_future_mask': recovery_mask,
            'future_tail_future_mask': future_tail_mask,
        }
        if flow_trace is not None:
            bridge_info['flow_trace'] = flow_trace
        return traj_final, bridge_info

    def _best_prior_mode(
        self,
        traj_prior: torch.Tensor,
        target_traj: torch.Tensor,
        loss_mask: torch.Tensor,
        agent_batch: Optional[torch.Tensor] = None,
        pairing: str = 'scene',
    ) -> torch.Tensor:
        prior_last = traj_prior[:, -1]  # [N, K, F, 2]
        errors = torch.norm(prior_last - target_traj.unsqueeze(1), p=2, dim=-1)
        masked_errors = errors * loss_mask.unsqueeze(1).float()
        per_agent = masked_errors.sum(dim=-1) / loss_mask.sum(dim=-1).clamp(min=1).unsqueeze(1)
        if pairing == 'agent':
            return per_agent.argmin(dim=-1)
        if pairing != 'scene':
            raise ValueError(f'Unknown flow mode pairing: {pairing}')
        if agent_batch is None:
            agent_batch = torch.zeros(traj_prior.size(0), dtype=torch.long, device=traj_prior.device)
        best = torch.zeros(traj_prior.size(0), dtype=torch.long, device=traj_prior.device)
        for scene_id in agent_batch.unique(sorted=True):
            scene = agent_batch == scene_id
            valid = loss_mask[scene].any(dim=-1)
            if valid.any():
                scene_scores = masked_errors[scene][valid].sum(dim=(0, 2))
                best[scene] = scene_scores.argmin()
        return best

    def flow_matching_loss(
        self,
        traj_prior: torch.Tensor,
        posterior_context: torch.Tensor,
        target_traj: torch.Tensor,
        recovery_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        future_tail_mask: torch.Tensor,
        history_traj: torch.Tensor,
        model_input_mask: torch.Tensor,
        train_best_mode_only: bool = True,
        agent_batch: Optional[torch.Tensor] = None,
        flow_mode_pairing: str = 'scene',
    ):
        num_agents = traj_prior.size(0)
        device = traj_prior.device
        target_traj = target_traj.to(device)
        history_traj = history_traj.to(device)
        recovery_mask = self._future_mask(recovery_mask, num_agents, device)
        loss_mask = self._future_mask(loss_mask, num_agents, device)
        future_tail_mask = self._future_mask(future_tail_mask, num_agents, device)
        model_input_future_mask = self._future_mask(model_input_mask, num_agents, device)
        history_endpoint = self._history_endpoint_from_tensors(history_traj, model_input_mask).to(device)

        best_mode = self._best_prior_mode(
            traj_prior, target_traj, loss_mask, agent_batch=agent_batch, pairing=flow_mode_pairing
        )
        agent_idx = torch.arange(num_agents, device=device)
        prior_last = traj_prior[:, -1]
        context_last = posterior_context[:, -1]

        if train_best_mode_only:
            prior_selected = prior_last[agent_idx, best_mode]  # [N, F, 2]
            context_selected = context_last[agent_idx, best_mode]  # [N, D]
            target_selected = target_traj
            recovery_selected = recovery_mask
            loss_selected = loss_mask
            tail_selected = future_tail_mask
            model_input_selected = model_input_future_mask
            history_selected = history_endpoint
        else:
            prior_selected = prior_last.reshape(num_agents * self.num_modes, self.num_future_steps, 2)
            context_selected = context_last.reshape(num_agents * self.num_modes, self.hidden_dim)
            target_selected = target_traj[:, None].expand(-1, self.num_modes, -1, -1).reshape_as(prior_selected)
            recovery_selected = recovery_mask[:, None].expand(-1, self.num_modes, -1).reshape(num_agents * self.num_modes, self.num_future_steps)
            loss_selected = loss_mask[:, None].expand(-1, self.num_modes, -1).reshape(num_agents * self.num_modes, self.num_future_steps)
            tail_selected = future_tail_mask[:, None].expand(-1, self.num_modes, -1).reshape(num_agents * self.num_modes, self.num_future_steps)
            model_input_selected = model_input_future_mask[:, None].expand(-1, self.num_modes, -1).reshape(num_agents * self.num_modes, self.num_future_steps)
            history_selected = history_endpoint[:, None].expand(-1, self.num_modes, -1).reshape(num_agents * self.num_modes, 2)

        noise = self.flow_train_sigma * torch.randn_like(prior_selected)
        x0 = prior_selected.detach() + noise
        x1 = target_selected
        t = torch.rand(x0.size(0), 1, 1, dtype=x0.dtype, device=device)
        x_t = (1.0 - t) * x0 + t * x1
        target_velocity = x1 - x0
        velocity = self._velocity_field(
            x_t=x_t,
            t=t,
            posterior_context=context_selected,
            recovery_mask=recovery_selected,
            future_tail_mask=tail_selected,
            model_input_mask=model_input_selected,
            history_endpoint=history_selected,
            observed_future=target_selected,
        )

        if loss_selected.any():
            flow_loss = (velocity - target_velocity).pow(2)[loss_selected].mean()
        else:
            flow_loss = velocity.sum() * 0.0
        flow_info = {
            'best_mode': best_mode,
            'loss_mask': loss_selected,
            'target_velocity': target_velocity.detach(),
            'pred_velocity': velocity.detach(),
        }
        return flow_loss, flow_info
