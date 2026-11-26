import torch
import torch.nn as nn

from utils import init_weights


class BridgeDecoder(nn.Module):
    """Generate a bounded posterior residual for missing-segment reconstruction."""

    def __init__(
        self,
        hidden_dim: int,
        num_historical_steps: int,
        num_future_steps: int,
        num_modes: int,
        residual_scale: float = 2.0,
        dropout: float = 0.1,
        use_latest_visible_history_anchor: bool = False,
    ) -> None:
        super(BridgeDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.num_modes = num_modes
        self.residual_scale = residual_scale
        self.use_latest_visible_history_anchor = use_latest_visible_history_anchor

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

    def _agent_tail_len(self, data, num_agents: int, device: torch.device) -> torch.Tensor:
        if 'tail_len' not in data:
            return torch.zeros(num_agents, dtype=torch.long, device=device)
        tail_len = data['tail_len'].to(device).long()
        if tail_len.dim() == 0:
            tail_len = tail_len.unsqueeze(0)
        if 'batch' in data['agent']:
            return tail_len[data['agent']['batch'].to(device)]
        return tail_len[0].expand(num_agents)

    def _build_gamma(self, data, posterior_valid: torch.Tensor, num_agents: int, device: torch.device) -> torch.Tensor:
        tail_len_agent = self._agent_tail_len(data, num_agents, device)
        missing_len_agent = torch.clamp(self.num_future_steps - tail_len_agent, min=0, max=self.num_future_steps)
        step_idx = torch.arange(self.num_future_steps, device=device).view(1, 1, 1, self.num_future_steps)

        missing_future_mask = step_idx < missing_len_agent.view(num_agents, 1, 1, 1)
        active = (tail_len_agent > 0).view(num_agents, 1, 1, 1) & posterior_valid.view(num_agents, 1, 1, 1)
        denom = missing_len_agent.clamp(min=1).float().view(num_agents, 1, 1, 1)
        gamma = (step_idx.float() + 1.0) / denom
        gamma = gamma * missing_future_mask.float() * active.float()
        gamma = gamma.unsqueeze(-1).expand(
            num_agents,
            self.num_historical_steps,
            self.num_modes,
            self.num_future_steps,
            1,
        ).contiguous()
        missing_future_mask = missing_future_mask.unsqueeze(-1).expand_as(gamma)
        return gamma, missing_future_mask

    def forward(self, traj_prior: torch.Tensor, posterior_context: torch.Tensor, posterior_valid: torch.Tensor, data):
        num_agents = traj_prior.size(0)
        device = traj_prior.device
        gamma, missing_future_mask = self._build_gamma(data, posterior_valid, num_agents, device)

        time_idx = torch.linspace(0.0, 1.0, self.num_future_steps, device=device)
        time_idx = time_idx.view(1, 1, 1, self.num_future_steps, 1).expand(num_agents, self.num_historical_steps, self.num_modes, -1, -1)
        distance_to_tail = 1.0 - gamma
        missing_flag = missing_future_mask.float()
        active_flag = (gamma > 0).float()

        context = posterior_context.unsqueeze(3).expand(-1, -1, -1, self.num_future_steps, -1)
        residual_input = torch.cat([context, time_idx, gamma, distance_to_tail, missing_flag], dim=-1)
        raw_residual = self.residual_mlp(residual_input)
        posterior_residual = self.residual_scale * torch.tanh(raw_residual)
        posterior_residual = posterior_residual * active_flag
        traj_final = traj_prior + gamma * posterior_residual

        bridge_info = {
            'posterior_residual': posterior_residual,
            'gamma': gamma,
            'missing_future_mask': missing_future_mask,
            'posterior_valid': posterior_valid,
        }
        return traj_final, bridge_info
