import torch
import torch.nn as nn

from datasets.history_anchor_utils import compute_latest_visible_history_state
from utils import init_weights
from utils import wrap_angle


class FutureTailEncoder(nn.Module):
    """Encode only observed future-tail states for reconstruction posterior context."""

    def __init__(
        self,
        hidden_dim: int,
        num_historical_steps: int,
        num_future_steps: int,
        num_modes: int,
        dropout: float = 0.1,
        use_latest_visible_history_anchor: bool = False,
    ) -> None:
        super(FutureTailEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.num_modes = num_modes
        self.use_latest_visible_history_anchor = use_latest_visible_history_anchor

        self.tail_mlp = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.apply(init_weights)

    def forward(self, data):
        position = data['agent']['position']
        heading = data['agent']['heading']
        model_input_mask = (
            data['agent']['model_input_mask']
            if 'model_input_mask' in data['agent']
            else data['agent']['visible_mask']
        )
        future_tail_mask = data['agent']['future_tail_mask']

        num_agents = position.size(0)
        device = position.device
        future_slice = slice(self.num_historical_steps, self.num_historical_steps + self.num_future_steps)

        future_pos = position[:, future_slice]
        future_heading = heading[:, future_slice]
        future_visible = model_input_mask[:, future_slice]
        tail_mask = future_tail_mask[:, future_slice] & future_visible

        if self.use_latest_visible_history_anchor:
            history_state = compute_latest_visible_history_state(data, self.num_historical_steps)
            last_hist_pos = history_state['anchor_pos'].unsqueeze(1)
            last_hist_heading = history_state['anchor_heading'].unsqueeze(1)
        else:
            hist_idx = max(self.num_historical_steps - 1, 0)
            last_hist_pos = position[:, hist_idx:hist_idx + 1]
            last_hist_heading = heading[:, hist_idx:hist_idx + 1]
        rel_tail_pos = future_pos - last_hist_pos
        heading_delta = wrap_angle(future_heading - last_hist_heading)

        time_idx = torch.linspace(0.0, 1.0, self.num_future_steps, device=device)
        time_idx = time_idx.view(1, self.num_future_steps, 1).expand(num_agents, -1, -1)
        mask_float = tail_mask.float().unsqueeze(-1)

        tail_input = torch.cat(
            [
                rel_tail_pos,
                time_idx,
                torch.sin(heading_delta).unsqueeze(-1),
                torch.cos(heading_delta).unsqueeze(-1),
                mask_float,
            ],
            dim=-1,
        )
        # Zeroing masked states is a defense-in-depth guard against missing-GT leakage.
        tail_input = tail_input * mask_float
        tail_emb = self.tail_mlp(tail_input) * mask_float
        valid_count = mask_float.sum(dim=1).clamp(min=1.0)
        posterior_agent_feature = tail_emb.sum(dim=1) / valid_count

        posterior_valid = tail_mask.any(dim=-1)
        posterior_agent_feature = torch.where(
            posterior_valid.unsqueeze(-1),
            self.output_norm(posterior_agent_feature),
            posterior_agent_feature.new_zeros(posterior_agent_feature.shape),
        )
        posterior_context = posterior_agent_feature[:, None, None, :].expand(
            num_agents,
            self.num_historical_steps,
            self.num_modes,
            self.hidden_dim,
        ).contiguous()
        return posterior_context, posterior_valid
