import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.utils import unbatch
import math

from losses import HuberTrajLoss
from losses import boundary_velocity_loss
from losses import endpoint_consistency_loss
from losses import jerk_smoothness_loss
from metrics import minJointADE
from metrics import minJointFDE
from metrics import acceleration_JS
from metrics import boundary_error
from metrics import gap_ADE
from metrics import gap_FDE
from metrics import relative_distance_JS
from metrics import velocity_JS
from modules import Backbone
from modules import MapEncoder
from modules import FutureTailEncoder
from modules import BridgeDecoder
from modules import ProgressiveBridgeDecoder
from modules import FlowBridgeDecoder
from datasets.history_anchor_utils import compute_latest_visible_history_state
from datasets.reconstruction_masks import build_target_agent_mask

from utils import generate_target
from utils import generate_predict_mask
from utils import compute_angles_lengths_2D
from utils import sigmoid_smooth_social_force, linear_smooth, social_force_smooth, sigmoid_smooth
from utils import compute_kinematics_from_traj
from utils import compute_relative_kinematics
from utils import js_divergence_histogram

class TrajBridge(pl.LightningModule):

    def __init__(self,
                 hidden_dim: int,
                 num_historical_steps: int,
                 num_future_steps: int,
                 duration: int,
                 a2a_radius: float,
                 l2a_radius: float,
                 num_visible_steps: int,
                 num_modes: int,
                 num_attn_layers: int,
                 num_hops: int,
                 num_heads: int,
                 dropout: float,
                 lr: float,
                 weight_decay: float,
                 warmup_epochs: int,
                 T_max: int,
                 **kwargs) -> None:
        super(TrajBridge, self).__init__()
        self.save_hyperparameters()
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.duration = duration
        self.a2a_radius = a2a_radius
        self.l2a_radius = l2a_radius
        self.num_visible_steps = num_visible_steps
        self.num_modes = num_modes
        self.num_attn_layers = num_attn_layers
        self.num_hops = num_hops
        self.num_heads = num_heads
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.T_max = T_max
        extra_kwargs = kwargs.get('kwargs', {})
        if isinstance(extra_kwargs, dict):
            extra_kwargs = {**extra_kwargs, **{k: v for k, v in kwargs.items() if k != 'kwargs'}}
        else:
            extra_kwargs = kwargs
        self.task_mode = extra_kwargs.get('task_mode', 'reconstruction')
        self.reconstruction_tail_lens = extra_kwargs.get('reconstruction_tail_lens', [0, 5, 10, 15])
        self.tail_len = extra_kwargs.get('tail_len', None)
        self.mixed_tail_len = extra_kwargs.get('mixed_tail_len', True)
        self.train_mixed_tail_len = extra_kwargs.get('train_mixed_tail_len', True)
        self.eval_tail_len = extra_kwargs.get('eval_tail_len', 5)
        self.eval_all_tail_lens = extra_kwargs.get('eval_all_tail_lens', False)
        self.map_encoder_mode = extra_kwargs.get('map_encoder_mode', 'auto')
        if self.map_encoder_mode == 'false':
            self.use_map_encoder = False
        elif self.map_encoder_mode == 'true':
            self.use_map_encoder = True
        else:
            self.use_map_encoder = extra_kwargs.get('use_map_encoder', True)
        self.use_boundary_loss = extra_kwargs.get('use_boundary_loss', False)
        self.boundary_loss_weight = extra_kwargs.get('boundary_loss_weight', 0.0)
        self.use_posterior_bridge = extra_kwargs.get('use_posterior_bridge', False)
        self.posterior_residual_scale = extra_kwargs.get('posterior_residual_scale', 2.0)
        self.posterior_dropout = extra_kwargs.get('posterior_dropout', 0.1)
        self.use_bridge_regularization = extra_kwargs.get('use_bridge_regularization', False)
        self.bridge_reg_weight = extra_kwargs.get('bridge_reg_weight', 0.0)
        self.bridge_decoder_type = extra_kwargs.get('bridge_decoder_type', 'none')
        if self.bridge_decoder_type in ('anchor', 'residual'):
            extra_kwargs['use_posterior_bridge'] = True
        elif self.bridge_decoder_type == 'progressive':
            extra_kwargs['use_progressive_bridge'] = True
        elif self.bridge_decoder_type == 'flow':
            extra_kwargs['use_flow_bridge'] = True
        self.use_posterior_bridge = extra_kwargs.get('use_posterior_bridge', self.use_posterior_bridge)
        self.use_progressive_bridge = extra_kwargs.get('use_progressive_bridge', False)
        self.use_flow_bridge = extra_kwargs.get('use_flow_bridge', False)
        self.num_bridge_stages = extra_kwargs.get('num_bridge_stages', 3)
        self.bridge_stage_order = extra_kwargs.get('bridge_stage_order', 'tail-side,middle,history-side')
        self.lambda_flow = extra_kwargs.get('lambda_flow', 0.0)
        self.flow_sigma = extra_kwargs.get('flow_sigma', 0.1)
        self.flow_train_sigma = extra_kwargs.get('flow_train_sigma', self.flow_sigma)
        self.flow_inference_sigma = extra_kwargs.get('flow_inference_sigma', 0.0)
        self.flow_inference_init = extra_kwargs.get('flow_inference_init', 'deterministic')
        self.flow_mode_pairing = extra_kwargs.get('flow_mode_pairing', 'scene')
        self.flow_ode_steps = extra_kwargs.get('flow_ode_steps', 8)
        self.flow_train_best_mode_only = extra_kwargs.get('flow_train_best_mode_only', True)
        self.lambda_boundary = extra_kwargs.get('lambda_boundary', 0.0)
        self.lambda_jerk = extra_kwargs.get('lambda_jerk', 0.0)
        self.use_recovery_smoothing = extra_kwargs.get('use_recovery_smoothing', False)
        self.use_latest_visible_history_anchor = extra_kwargs.get('use_latest_visible_history_anchor', False)
        self.use_agent_interaction = extra_kwargs.get('use_agent_interaction', True)
        self.use_temporal_interaction = extra_kwargs.get('use_temporal_interaction', True)
        self.use_mode_interaction = extra_kwargs.get('use_mode_interaction', True)
        self.target_agent_policy = extra_kwargs.get('target_agent_policy', 'category')
        self.target_categories = tuple(extra_kwargs.get('target_categories', [1]))

        self.Backbone = Backbone(
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            num_future_steps=num_future_steps,
            duration=duration,
            a2a_radius=a2a_radius,
            l2a_radius=l2a_radius,
            num_attn_layers=num_attn_layers,
            num_modes=num_modes,
            num_heads=num_heads,
            dropout=dropout,
            use_agent_interaction=self.use_agent_interaction,
            use_temporal_interaction=self.use_temporal_interaction,
            use_mode_interaction=self.use_mode_interaction,
        )
        self.MapEncoder = None
        if self.use_map_encoder:
            self.MapEncoder = MapEncoder(
                hidden_dim=hidden_dim,
                num_hops=num_hops,
                num_heads=num_heads,
                dropout=dropout
            )
        self.FutureTailEncoder = None
        self.BridgeDecoder = None
        bridge_requested = (
            (self.task_mode == 'reconstruction' and self.use_posterior_bridge)
            or (
                self.task_mode == 'intermittent_recovery'
                and (self.use_posterior_bridge or self.use_progressive_bridge or self.use_flow_bridge)
            )
        )
        if bridge_requested:
            self.FutureTailEncoder = FutureTailEncoder(
                hidden_dim=hidden_dim,
                num_historical_steps=num_historical_steps,
                num_future_steps=num_future_steps,
                num_modes=num_modes,
                dropout=self.posterior_dropout,
                use_latest_visible_history_anchor=self.use_latest_visible_history_anchor,
            )
            if self.task_mode == 'intermittent_recovery' and self.use_flow_bridge:
                self.BridgeDecoder = FlowBridgeDecoder(
                    hidden_dim=hidden_dim,
                    num_historical_steps=num_historical_steps,
                    num_future_steps=num_future_steps,
                    num_modes=num_modes,
                    flow_sigma=self.flow_sigma,
                    flow_train_sigma=self.flow_train_sigma,
                    flow_inference_sigma=self.flow_inference_sigma,
                    flow_inference_init=self.flow_inference_init,
                    flow_ode_steps=self.flow_ode_steps,
                    dropout=self.posterior_dropout,
                    use_latest_visible_history_anchor=self.use_latest_visible_history_anchor,
                )
            elif self.task_mode == 'intermittent_recovery' and self.use_progressive_bridge:
                self.BridgeDecoder = ProgressiveBridgeDecoder(
                    hidden_dim=hidden_dim,
                    num_historical_steps=num_historical_steps,
                    num_future_steps=num_future_steps,
                    num_modes=num_modes,
                    num_bridge_stages=self.num_bridge_stages,
                    bridge_stage_order=self.bridge_stage_order,
                    residual_scale=self.posterior_residual_scale,
                    dropout=self.posterior_dropout,
                    use_latest_visible_history_anchor=self.use_latest_visible_history_anchor,
                )
            else:
                self.BridgeDecoder = BridgeDecoder(
                    hidden_dim=hidden_dim,
                    num_historical_steps=num_historical_steps,
                    num_future_steps=num_future_steps,
                    num_modes=num_modes,
                    residual_scale=self.posterior_residual_scale,
                    dropout=self.posterior_dropout,
                    use_latest_visible_history_anchor=self.use_latest_visible_history_anchor,
                )
        self._last_bridge_info = None
        self._last_bridge_debug = {}

        self.reg_loss_traj = HuberTrajLoss()

        self.min_joint_ade = minJointADE()
        self.min_joint_fde = minJointFDE()
    def _target_agent_mask(self, data: Batch) -> torch.Tensor:
        return build_target_agent_mask(data, self.target_agent_policy, self.target_categories)

    def _has_map_inputs(self, data: Batch) -> bool:
        try:
            return (
                'position' in data['lane']
                and 'position' in data['polyline']
                and 'polyline_to_lane_edge_index' in data['polyline', 'lane']
                and 'successor_edge_index' in data['lane', 'lane']
            )
        except (KeyError, AttributeError):
            return False

    def _is_recovery_task(self) -> bool:
        return self.task_mode in ('reconstruction', 'intermittent_recovery')

    def _agent_timeline_mask(self, data: Batch, key: str, fallback_key: str = 'visible_mask') -> torch.Tensor:
        if key in data['agent']:
            return data['agent'][key]
        return data['agent'][fallback_key]

    def _sync_latest_visible_anchor_flag(self) -> None:
        for module in (self.FutureTailEncoder, self.BridgeDecoder):
            if module is not None and hasattr(module, 'use_latest_visible_history_anchor'):
                module.use_latest_visible_history_anchor = self.use_latest_visible_history_anchor

    def _history_anchor_state(self, data: Batch) -> dict:
        if self.use_latest_visible_history_anchor:
            return compute_latest_visible_history_state(data, self.num_historical_steps)

        position = data['agent']['position']
        heading = data['agent']['heading']
        hist_idx = max(self.num_historical_steps - 1, 0)
        prev_idx = max(hist_idx - 1, 0)
        num_agents = position.size(0)
        device = position.device
        agent_idx = torch.arange(num_agents, device=device)
        anchor_pos = position[:, hist_idx]
        prev_pos = position[:, prev_idx]
        prev_valid = torch.full((num_agents,), hist_idx > 0, dtype=torch.bool, device=device)
        anchor_velocity = torch.where(
            prev_valid.unsqueeze(-1),
            anchor_pos - prev_pos,
            torch.zeros_like(anchor_pos),
        )
        return {
            'anchor_pos': anchor_pos,
            'anchor_heading': heading[:, hist_idx],
            'anchor_index': torch.full((num_agents,), hist_idx, dtype=torch.long, device=device),
            'anchor_valid': torch.ones(num_agents, dtype=torch.bool, device=device),
            'prev_pos': prev_pos,
            'prev_index': torch.full((num_agents,), prev_idx, dtype=torch.long, device=device),
            'prev_valid': prev_valid,
            'anchor_velocity': anchor_velocity,
            'delta_t': torch.ones(num_agents, dtype=position.dtype, device=device),
        }

    def _align_anchor_future(self, selected: torch.Tensor, anchor_index: torch.Tensor) -> torch.Tensor:
        if not self.use_latest_visible_history_anchor:
            return selected
        if selected.size(-2) != self.num_future_steps:
            return selected

        num_agents = selected.size(0)
        device = selected.device
        offset = (self.num_historical_steps - 1 - anchor_index.to(device)).clamp(min=0, max=self.num_future_steps - 1)
        source_idx = torch.arange(self.num_future_steps, device=device).view(1, -1) + offset.view(num_agents, 1)
        valid = source_idx < self.num_future_steps
        clamped = source_idx.clamp(max=self.num_future_steps - 1).long()

        if selected.dim() == 4:
            gather_index = clamped[:, None, :, None].expand(-1, selected.size(1), -1, selected.size(-1))
            aligned = selected.gather(dim=2, index=gather_index)
            if valid.all() or self.num_future_steps < 2:
                return aligned
            last = selected[:, :, -1]
            prev = selected[:, :, -2]
            velocity = last - prev
            extra = (source_idx - (self.num_future_steps - 1)).clamp(min=0).to(selected.dtype)
            extrapolated = last[:, :, None] + extra[:, None, :, None] * velocity[:, :, None]
            return torch.where(valid[:, None, :, None], aligned, extrapolated)

        if selected.dim() == 3:
            gather_index = clamped[:, :, None].expand(-1, -1, selected.size(-1))
            aligned = selected.gather(dim=1, index=gather_index)
            if valid.all() or self.num_future_steps < 2:
                return aligned
            last = selected[:, -1]
            prev = selected[:, -2]
            velocity = last - prev
            extra = (source_idx - (self.num_future_steps - 1)).clamp(min=0).to(selected.dtype)
            extrapolated = last[:, None] + extra[:, :, None] * velocity[:, None]
            return torch.where(valid[:, :, None], aligned, extrapolated)

        raise ValueError(f'Unsupported selected trajectory shape: {tuple(selected.shape)}')

    def _select_reconstruction_anchor_predictions(
        self,
        data: Batch,
        traj: torch.Tensor,
        agent_mask: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_latest_visible_history_anchor:
            return traj[agent_mask, -1]
        anchor_state = self._history_anchor_state(data)
        anchor_index = anchor_state['anchor_index'][agent_mask].clamp(min=0, max=self.num_historical_steps - 1)
        selected = traj[agent_mask]
        local_agent_idx = torch.arange(selected.size(0), device=selected.device)
        selected = selected[local_agent_idx, anchor_index]
        return self._align_anchor_future(selected, anchor_index)

    def forward(self,
                data: Batch):
        self._sync_latest_visible_anchor_flag()
        lane_embs = None
        if self.use_map_encoder and self.MapEncoder is not None and self._has_map_inputs(data):
            lane_embs = self.MapEncoder(data=data)

        # The posterior bridge refines the history/map prior using the observed tail.
        traj_propose, traj_output = self.Backbone(data=data, l_embs=lane_embs)
        self._last_bridge_info = None
        self._last_bridge_debug = {}

        if (
            self._is_recovery_task()
            and self.FutureTailEncoder is not None
            and self.BridgeDecoder is not None
        ):
            traj_prior = traj_output
            posterior_context, posterior_valid = self.FutureTailEncoder(data)
            traj_output, bridge_info = self.BridgeDecoder(
                traj_prior=traj_prior,
                posterior_context=posterior_context,
                posterior_valid=posterior_valid,
                data=data,
            )
            bridge_info['posterior_context'] = posterior_context
            bridge_info['traj_prior'] = traj_prior
            bridge_info['traj_final'] = traj_output
            self._last_bridge_info = bridge_info
            self._last_bridge_debug = {
                key: value.detach() if isinstance(value, torch.Tensor) else value
                for key, value in bridge_info.items()
            }

        return traj_propose, traj_output

    def _slice_future(self, tensor: torch.Tensor) -> torch.Tensor:
        future = tensor[:, self.num_historical_steps:self.num_historical_steps + self.num_future_steps]
        if future.size(1) == self.num_future_steps:
            return future
        pad_shape = list(future.shape)
        pad_shape[1] = self.num_future_steps - future.size(1)
        return torch.cat([future, future.new_zeros(pad_shape)], dim=1)

    def _last_valid_indices(self, mask: torch.Tensor):
        valid = mask.any(dim=-1)
        steps = torch.arange(mask.size(-1), device=mask.device).unsqueeze(0).expand_as(mask)
        last_idx = torch.where(mask, steps, torch.zeros_like(steps)).max(dim=-1).values
        return last_idx.long(), valid

    def _get_reconstruction_eval_data(self, data: Batch, traj_output: torch.Tensor):
        agent_mask = self._target_agent_mask(data)
        pred_modes = self._select_reconstruction_anchor_predictions(data, traj_output, agent_mask)
        target_traj = self._slice_future(data['agent']['position'])[agent_mask]
        if 'loss_mask' in data['agent']:
            future_loss_mask = self._slice_future(data['agent']['loss_mask'])[agent_mask]
        else:
            future_loss_mask = self._slice_future(data['agent']['visible_mask'])[agent_mask]
        agent_batch = data['agent']['batch'][agent_mask]
        return agent_mask, pred_modes, target_traj, future_loss_mask, agent_batch

    def _evaluate_reconstruction_predictions(
        self,
        pred_modes: torch.Tensor,
        target_traj: torch.Tensor,
        future_loss_mask: torch.Tensor,
        agent_batch: torch.Tensor,
        batch_size: int,
    ):
        device = pred_modes.device
        zero = pred_modes.sum() * 0.0
        metrics = {
            'ADE': zero,
            'FDE': zero,
            'minJointADE': zero,
            'minJointFDE': zero,
        }
        num_agents = pred_modes.size(0)
        if num_agents == 0:
            return metrics, pred_modes.new_zeros((0, self.num_future_steps, 2)), torch.zeros(0, dtype=torch.bool, device=device)

        errors = torch.norm(pred_modes - target_traj.unsqueeze(1), p=2, dim=-1)  # [N, K, F]
        valid_agent = future_loss_mask.any(dim=-1)
        if not valid_agent.any():
            return metrics, pred_modes[:, 0], valid_agent

        valid_counts = future_loss_mask.sum(dim=-1).clamp(min=1)
        mode_error_sums = (errors * future_loss_mask.unsqueeze(1)).sum(dim=-1)
        per_agent_ade = mode_error_sums / valid_counts.unsqueeze(1)
        per_agent_best_mode = per_agent_ade.argmin(dim=-1)
        agent_idx = torch.arange(num_agents, device=device)
        best_agent_pred = pred_modes[agent_idx, per_agent_best_mode]
        best_agent_errors = errors[agent_idx, per_agent_best_mode]

        last_idx, _ = self._last_valid_indices(future_loss_mask)
        metrics['ADE'] = best_agent_errors[future_loss_mask].mean()
        metrics['FDE'] = best_agent_errors[agent_idx[valid_agent], last_idx[valid_agent]].mean()

        joint_best_mode = torch.zeros(num_agents, dtype=torch.long, device=device)
        joint_ades = []
        joint_fdes = []
        for batch_idx in range(batch_size):
            scene_mask = (agent_batch == batch_idx) & valid_agent
            if not scene_mask.any():
                continue
            scene_errors = errors[scene_mask]
            scene_loss_mask = future_loss_mask[scene_mask]
            scene_mode_sums = (scene_errors * scene_loss_mask.unsqueeze(1)).sum(dim=(0, 2))
            best_mode = scene_mode_sums.argmin()
            joint_best_mode[scene_mask] = best_mode
            joint_ades.append(scene_mode_sums[best_mode] / scene_loss_mask.sum().clamp(min=1))
            scene_last_idx, scene_valid = self._last_valid_indices(scene_loss_mask)
            scene_agent_idx = torch.arange(scene_errors.size(0), device=device)
            joint_fdes.append(scene_errors[scene_agent_idx[scene_valid], best_mode, scene_last_idx[scene_valid]].mean())

        if joint_ades:
            metrics['minJointADE'] = torch.stack(joint_ades).mean()
            metrics['minJointFDE'] = torch.stack(joint_fdes).mean()

        best_joint_pred = pred_modes[agent_idx, joint_best_mode]
        return metrics, best_joint_pred, valid_agent

    def _compute_reconstruction_metrics(self, data: Batch, traj_output: torch.Tensor, prefix: str):
        agent_mask, pred_modes, target_traj, future_loss_mask, agent_batch = self._get_reconstruction_eval_data(data, traj_output)
        metrics, best_joint_pred, valid_agent = self._evaluate_reconstruction_predictions(
            pred_modes=pred_modes,
            target_traj=target_traj,
            future_loss_mask=future_loss_mask,
            agent_batch=agent_batch,
            batch_size=len(data['case_id']),
        )
        for name, value in metrics.items():
            self.log(f'{prefix}_recon_{name}', value, prog_bar=name in ('ADE', 'FDE', 'minJointFDE'),
                     on_step=False, on_epoch=True, batch_size=len(data['case_id']), sync_dist=True)
        recovery_metric_values = {}
        if self.task_mode == 'intermittent_recovery':
            recovery_metric_values = {
                'gap_ADE': gap_ADE(best_joint_pred, target_traj, future_loss_mask),
                'gap_FDE': gap_FDE(best_joint_pred, target_traj, future_loss_mask),
                'boundary_error': boundary_error(best_joint_pred, target_traj, future_loss_mask),
                'velocity_JS': velocity_JS(best_joint_pred, target_traj, future_loss_mask),
                'acceleration_JS': acceleration_JS(best_joint_pred, target_traj, future_loss_mask),
                'relative_distance_JS': relative_distance_JS(best_joint_pred, target_traj, future_loss_mask, agent_batch),
            }
            for name, value in recovery_metric_values.items():
                self.log(
                    f'{prefix}_recovery_{name}',
                    value,
                    prog_bar=name in ('gap_ADE', 'gap_FDE'),
                    on_step=False,
                    on_epoch=True,
                    batch_size=len(data['case_id']),
                    sync_dist=True,
                )
        return {
            'agent_mask': agent_mask,
            'pred_modes': pred_modes,
            'target_traj': target_traj,
            'future_loss_mask': future_loss_mask,
            'agent_batch': agent_batch,
            'metrics': metrics,
            'recovery_metrics': recovery_metric_values,
            'best_joint_pred': best_joint_pred,
            'valid_agent': valid_agent,
        }

    def _masked_reg_loss(self, predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.any():
            return self.reg_loss_traj(predictions[mask], targets[mask])
        return predictions.sum() * 0.0

    def _compute_prediction_loss(self, data: Batch, traj_propose: torch.Tensor, traj_output: torch.Tensor):
        agent_mask = self._target_agent_mask(data)
        traj_propose = traj_propose[agent_mask]
        traj_output = traj_output[agent_mask]

        target_traj, target_mask = generate_target(position=data['agent']['position'],
                                                   mask=data['agent']['visible_mask'],
                                                   num_historical_steps=self.num_historical_steps,
                                                   num_future_steps=self.num_future_steps)
        target_traj = target_traj[agent_mask]
        target_mask = target_mask[agent_mask]

        agent_batch = data['agent']['batch'][agent_mask]
        errors = (torch.norm(traj_propose - target_traj.unsqueeze(2), p=2, dim=-1) * target_mask.unsqueeze(2)).sum(dim=-1)
        joint_errors = [error.sum(dim=0, keepdim=True) for error in unbatch(errors, agent_batch)]
        joint_errors = torch.cat(joint_errors, dim=0)

        num_agent_pre_batch = torch.bincount(agent_batch)
        best_mode_index = joint_errors.argmin(dim=-1)
        best_mode_index = best_mode_index.repeat_interleave(num_agent_pre_batch, 0)
        agent_idx = torch.arange(traj_propose.size(0), device=traj_propose.device)[:, None]
        hist_idx = torch.arange(traj_propose.size(1), device=traj_propose.device)[None, :]
        traj_best_propose = traj_propose[agent_idx, hist_idx, best_mode_index]
        traj_best_output = traj_output[agent_idx, hist_idx, best_mode_index]

        predict_mask = generate_predict_mask(data['agent']['visible_mask'][agent_mask, :self.num_historical_steps], self.num_visible_steps)
        targ_mask = target_mask[predict_mask]
        traj_pro = traj_best_propose[predict_mask]
        traj_ref = traj_best_output[predict_mask]
        targ_traj = target_traj[predict_mask]

        reg_loss_traj_propose = self._masked_reg_loss(traj_pro, targ_traj, targ_mask)
        reg_loss_traj_refine = self._masked_reg_loss(traj_ref, targ_traj, targ_mask)
        return reg_loss_traj_propose, reg_loss_traj_refine, traj_best_output

    def _compute_boundary_loss(
        self,
        traj: torch.Tensor,
        data: Batch,
        agent_mask: torch.Tensor,
        agent_batch: torch.Tensor,
        future_loss_mask: torch.Tensor,
    ) -> torch.Tensor:
        position = data['agent']['position'][agent_mask]
        visible_mask = self._agent_timeline_mask(data, 'gt_visible_mask')[agent_mask]
        anchor_state = self._history_anchor_state(data)
        anchor_pos = anchor_state['anchor_pos'][agent_mask]
        anchor_velocity = anchor_state['anchor_velocity'][agent_mask]
        anchor_valid = anchor_state['anchor_valid'][agent_mask]
        losses = []

        if self.num_historical_steps >= 2 and self.num_future_steps > 0:
            if self.use_latest_visible_history_anchor:
                hist_vel = anchor_velocity
                pred_start_vel = traj[:, 0] - anchor_pos
                start_mask = future_loss_mask[:, 0] & anchor_valid
            else:
                hist_vel = position[:, self.num_historical_steps - 1] - position[:, self.num_historical_steps - 2]
                pred_start_vel = traj[:, 0] - position[:, self.num_historical_steps - 1]
                start_mask = (
                    future_loss_mask[:, 0]
                    & visible_mask[:, self.num_historical_steps - 1]
                    & visible_mask[:, self.num_historical_steps - 2]
                )
            if start_mask.any():
                losses.append(F.smooth_l1_loss(pred_start_vel[start_mask], hist_vel[start_mask], reduction='none').sum(-1).mean())

        if 'tail_len' in data and self.num_future_steps > 0:
            tail_len = data['tail_len'].to(agent_batch.device).long()
            if tail_len.dim() == 0:
                tail_len = tail_len.unsqueeze(0)
            tail_len_agent = tail_len[agent_batch]
            missing_len_agent = torch.clamp(self.num_future_steps - tail_len_agent, min=0, max=self.num_future_steps)
            agent_idx = torch.arange(traj.size(0), device=traj.device)
            last_missing_idx = (missing_len_agent - 1).clamp(min=0)
            prev_missing_idx = (last_missing_idx - 1).clamp(min=0)
            tail_first_idx = self.num_historical_steps + missing_len_agent
            valid_tail_idx = tail_first_idx < position.size(1)
            tail_first_idx = tail_first_idx.clamp(max=position.size(1) - 1)

            pred_end = traj[agent_idx, last_missing_idx]
            pred_prev = traj[agent_idx, prev_missing_idx]
            hist_last = anchor_pos if self.use_latest_visible_history_anchor else position[:, self.num_historical_steps - 1]
            pred_prev = torch.where((missing_len_agent > 1).unsqueeze(-1), pred_prev, hist_last)
            tail_first = position[agent_idx, tail_first_idx]
            end_mask = (
                (tail_len_agent > 0)
                & (missing_len_agent > 0)
                & valid_tail_idx
                & future_loss_mask[agent_idx, last_missing_idx]
                & visible_mask[agent_idx, tail_first_idx]
            )
            if end_mask.any():
                pred_end_vel = pred_end[end_mask] - pred_prev[end_mask]
                tail_entry_vel = tail_first[end_mask] - pred_end[end_mask]
                losses.append(F.smooth_l1_loss(pred_end_vel, tail_entry_vel, reduction='none').sum(-1).mean())

        if losses:
            return torch.stack(losses).sum()
        return traj.sum() * 0.0

    def _compute_reconstruction_loss(self, data: Batch, traj_propose: torch.Tensor, traj_output: torch.Tensor):
        agent_mask = self._target_agent_mask(data)
        traj_propose = self._select_reconstruction_anchor_predictions(data, traj_propose, agent_mask)
        traj_output = self._select_reconstruction_anchor_predictions(data, traj_output, agent_mask)

        target_traj = self._slice_future(data['agent']['position'])[agent_mask]
        if 'loss_mask' in data['agent']:
            future_loss_mask = self._slice_future(data['agent']['loss_mask'])[agent_mask]
        else:
            future_loss_mask = self._slice_future(data['agent']['visible_mask'])[agent_mask]

        agent_batch = data['agent']['batch'][agent_mask]
        errors = (torch.norm(traj_propose - target_traj.unsqueeze(1), p=2, dim=-1) * future_loss_mask.unsqueeze(1)).sum(dim=-1)
        joint_errors = [error.sum(dim=0, keepdim=True) for error in unbatch(errors, agent_batch)]
        joint_errors = torch.cat(joint_errors, dim=0)

        num_agent_pre_batch = torch.bincount(agent_batch)
        best_mode_index = joint_errors.argmin(dim=-1).repeat_interleave(num_agent_pre_batch, 0)
        agent_idx = torch.arange(traj_propose.size(0), device=traj_propose.device)
        traj_best_propose = traj_propose[agent_idx, best_mode_index]
        traj_best_output = traj_output[agent_idx, best_mode_index]

        reg_loss_traj_propose = self._masked_reg_loss(traj_best_propose, target_traj, future_loss_mask)
        reg_loss_traj_refine = self._masked_reg_loss(traj_best_output, target_traj, future_loss_mask)
        boundary_loss = traj_best_output.sum() * 0.0
        if self.use_boundary_loss and self.boundary_loss_weight > 0.0:
            boundary_loss = self._compute_boundary_loss(
                traj=traj_best_output,
                data=data,
                agent_mask=agent_mask,
                agent_batch=agent_batch,
                future_loss_mask=future_loss_mask,
            )
        return reg_loss_traj_propose, reg_loss_traj_refine, traj_best_output, boundary_loss

    def _compute_recovery_aux_losses(self, data: Batch, traj_best_output: torch.Tensor) -> dict:
        zero = traj_best_output.sum() * 0.0
        losses = {
            'boundary_velocity': zero,
            'endpoint_consistency': zero,
            'jerk': zero,
        }
        if self.task_mode != 'intermittent_recovery':
            return losses
        if self.lambda_boundary <= 0.0 and self.lambda_jerk <= 0.0:
            return losses

        agent_mask = self._target_agent_mask(data)
        target_full = data['agent']['position'][agent_mask]
        target_future = self._slice_future(data['agent']['position'])[agent_mask]
        gt_visible_mask = self._agent_timeline_mask(data, 'gt_visible_mask')[agent_mask]
        model_input_mask = self._agent_timeline_mask(data, 'model_input_mask')[agent_mask]
        recovery_mask = self._agent_timeline_mask(data, 'recovery_mask', fallback_key='missing_mask')[agent_mask]
        future_tail_mask = data['agent']['future_tail_mask'][agent_mask] if 'future_tail_mask' in data['agent'] else torch.zeros_like(recovery_mask)
        loss_mask = self._slice_future(data['agent']['loss_mask'])[agent_mask] if 'loss_mask' in data['agent'] else self._slice_future(recovery_mask & gt_visible_mask)

        if self.lambda_boundary > 0.0:
            losses['boundary_velocity'] = boundary_velocity_loss(
                pred_future=traj_best_output,
                target_full=target_full,
                gt_visible_mask=gt_visible_mask,
                history_len=self.num_historical_steps,
                recovery_mask=recovery_mask,
                future_tail_mask=future_tail_mask,
            )
            losses['endpoint_consistency'] = endpoint_consistency_loss(
                pred_future=traj_best_output,
                target_future=target_future,
                loss_mask=loss_mask,
            )

        if self.lambda_jerk > 0.0:
            full_pred = target_full.clone()
            future_end = min(self.num_historical_steps + self.num_future_steps, full_pred.size(1))
            future_len = max(0, future_end - self.num_historical_steps)
            if future_len > 0:
                recovery_future = recovery_mask[:, self.num_historical_steps:future_end]
                pred_slice = traj_best_output[:, :future_len]
                current_slice = full_pred[:, self.num_historical_steps:future_end]
                full_pred[:, self.num_historical_steps:future_end] = torch.where(
                    recovery_future.unsqueeze(-1),
                    pred_slice,
                    current_slice,
                )
            valid_mask = model_input_mask | recovery_mask
            losses['jerk'] = jerk_smoothness_loss(
                full_traj=full_pred,
                valid_mask=valid_mask,
                recovery_mask=recovery_mask,
            )

        return losses

    def _compute_flow_bridge_loss(self, data: Batch, reference: torch.Tensor) -> torch.Tensor:
        if not (self.task_mode == 'intermittent_recovery' and self.use_flow_bridge and self.lambda_flow > 0.0):
            return reference.sum() * 0.0
        if self._last_bridge_info is None or 'traj_prior' not in self._last_bridge_info or 'posterior_context' not in self._last_bridge_info:
            return reference.sum() * 0.0
        if self.BridgeDecoder is None or not hasattr(self.BridgeDecoder, 'flow_matching_loss'):
            return reference.sum() * 0.0

        agent_mask = self._target_agent_mask(data)
        if not agent_mask.any():
            return reference.sum() * 0.0

        traj_prior = self._last_bridge_info['traj_prior'][agent_mask]  # [N, H, K, F, 2]
        posterior_context = self._last_bridge_info['posterior_context'][agent_mask]  # [N, H, K, D]
        target_traj = self._slice_future(data['agent']['position'])[agent_mask]  # [N, F, 2]
        recovery_mask = self._agent_timeline_mask(data, 'recovery_mask', fallback_key='missing_mask')[agent_mask]
        loss_mask = data['agent']['loss_mask'][agent_mask] if 'loss_mask' in data['agent'] else recovery_mask
        future_tail_mask = data['agent']['future_tail_mask'][agent_mask] if 'future_tail_mask' in data['agent'] else torch.zeros_like(recovery_mask)
        model_input_mask = self._agent_timeline_mask(data, 'model_input_mask')[agent_mask]
        history_traj = data['agent']['position'][agent_mask, :self.num_historical_steps]

        flow_loss, flow_info = self.BridgeDecoder.flow_matching_loss(
            traj_prior=traj_prior,
            posterior_context=posterior_context,
            target_traj=target_traj,
            recovery_mask=recovery_mask,
            loss_mask=loss_mask,
            future_tail_mask=future_tail_mask,
            history_traj=history_traj,
            model_input_mask=model_input_mask,
            train_best_mode_only=self.flow_train_best_mode_only,
            agent_batch=data['agent']['batch'][agent_mask],
            flow_mode_pairing=self.flow_mode_pairing,
        )
        self._last_bridge_info['flow_info'] = flow_info
        return flow_loss

    def _compute_bridge_regularization_loss(self, data: Batch, reference: torch.Tensor) -> torch.Tensor:
        if not (self.use_bridge_regularization and self.bridge_reg_weight > 0.0):
            return reference.sum() * 0.0
        if self._last_bridge_info is None or 'posterior_residual' not in self._last_bridge_info:
            return reference.sum() * 0.0

        residual = self._last_bridge_info['posterior_residual']
        agent_mask = self._target_agent_mask(data)
        if 'recovery_mask' in data['agent']:
            missing_mask = self._slice_future(data['agent']['recovery_mask'])[agent_mask]
        elif 'missing_mask' in data['agent']:
            missing_mask = self._slice_future(data['agent']['missing_mask'])[agent_mask]
        else:
            missing_mask = self._slice_future(data['agent']['visible_mask'])[agent_mask]
        residual = residual[agent_mask, -1]
        if missing_mask.any():
            return residual[missing_mask].pow(2).sum(dim=-1).mean()
        return residual.sum() * 0.0

    def training_step(self,data,batch_idx):
        traj_propose, traj_output = self(data)               # [N, H, K, F, 2]

        if self._is_recovery_task():
            reg_loss_traj_propose, reg_loss_traj_refine, traj_best_output, boundary_loss = self._compute_reconstruction_loss(data, traj_propose, traj_output)
            bridge_reg_loss = self._compute_bridge_regularization_loss(data, traj_output)
            flow_loss = self._compute_flow_bridge_loss(data, traj_best_output)
            recovery_aux_losses = self._compute_recovery_aux_losses(data, traj_best_output)
            recovery_boundary_loss = recovery_aux_losses['boundary_velocity'] + recovery_aux_losses['endpoint_consistency']
            loss = (
                reg_loss_traj_propose
                + reg_loss_traj_refine
                + self.lambda_flow * flow_loss
                + self.boundary_loss_weight * boundary_loss
                + self.bridge_reg_weight * bridge_reg_loss
                + self.lambda_boundary * recovery_boundary_loss
                + self.lambda_jerk * recovery_aux_losses['jerk']
            )
            self.log('train_boundary_loss', boundary_loss, prog_bar=False, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
            self.log('train_bridge_reg_loss', bridge_reg_loss, prog_bar=False, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
            self.log('train_flow_loss', flow_loss, prog_bar=False, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
            self.log('train_recovery_boundary_velocity_loss', recovery_aux_losses['boundary_velocity'], prog_bar=False, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
            self.log('train_recovery_endpoint_loss', recovery_aux_losses['endpoint_consistency'], prog_bar=False, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
            self.log('train_recovery_jerk_loss', recovery_aux_losses['jerk'], prog_bar=False, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
            self.log('loss_reconstruction', reg_loss_traj_refine, prog_bar=False, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
            self.log('loss_flow', flow_loss, prog_bar=False, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
            self.log('loss_boundary', recovery_boundary_loss, prog_bar=False, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
            self.log('loss_jerk', recovery_aux_losses['jerk'], prog_bar=False, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
            self.log('loss_total', loss, prog_bar=False, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
        else:
            reg_loss_traj_propose, reg_loss_traj_refine, _ = self._compute_prediction_loss(data, traj_propose, traj_output)
            loss = reg_loss_traj_propose + reg_loss_traj_refine
        self.log('train_reg_loss_traj_propose', reg_loss_traj_propose, prog_bar=True, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('train_reg_loss_traj_refine', reg_loss_traj_refine, prog_bar=True, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)

        return loss

    def validation_step(self,data,batch_idx):
        traj_propose, traj_output = self(data)               #[(N1,...,Nb),H,K,F,2],[(N1,...,Nb),H,K,F,2]

        if self._is_recovery_task():
            reg_loss_traj_propose, reg_loss_traj_refine, traj_best_output, boundary_loss = self._compute_reconstruction_loss(data, traj_propose, traj_output)
            bridge_reg_loss = self._compute_bridge_regularization_loss(data, traj_output)
            flow_loss = self._compute_flow_bridge_loss(data, traj_best_output)
            recovery_aux_losses = self._compute_recovery_aux_losses(data, traj_best_output)
            recovery_boundary_loss = recovery_aux_losses['boundary_velocity'] + recovery_aux_losses['endpoint_consistency']
            loss = (
                reg_loss_traj_propose
                + reg_loss_traj_refine
                + self.lambda_flow * flow_loss
                + self.boundary_loss_weight * boundary_loss
                + self.bridge_reg_weight * bridge_reg_loss
                + self.lambda_boundary * recovery_boundary_loss
                + self.lambda_jerk * recovery_aux_losses['jerk']
            )
            self.log('val_boundary_loss', boundary_loss, prog_bar=False, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
            self.log('val_bridge_reg_loss', bridge_reg_loss, prog_bar=False, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
            self.log('val_flow_loss', flow_loss, prog_bar=False, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
            self.log('val_recovery_boundary_velocity_loss', recovery_aux_losses['boundary_velocity'], prog_bar=False, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
            self.log('val_recovery_endpoint_loss', recovery_aux_losses['endpoint_consistency'], prog_bar=False, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
            self.log('val_recovery_jerk_loss', recovery_aux_losses['jerk'], prog_bar=False, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
            self.log('loss_reconstruction', reg_loss_traj_refine, prog_bar=False, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
            self.log('loss_flow', flow_loss, prog_bar=False, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
            self.log('loss_boundary', recovery_boundary_loss, prog_bar=False, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
            self.log('loss_jerk', recovery_aux_losses['jerk'], prog_bar=False, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
            self.log('loss_total', loss, prog_bar=False, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
        else:
            reg_loss_traj_propose, reg_loss_traj_refine, _ = self._compute_prediction_loss(data, traj_propose, traj_output)
            loss = reg_loss_traj_propose + reg_loss_traj_refine
        self.log('val_reg_loss_traj_propose', reg_loss_traj_propose, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('val_reg_loss_traj_refine', reg_loss_traj_refine, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)

        if self._is_recovery_task():
            self._compute_reconstruction_metrics(data=data, traj_output=traj_output, prefix='val')
            return

        agent_mask = self._target_agent_mask(data)
        traj_output = traj_output[agent_mask]
        target_traj, target_mask = generate_target(position=data['agent']['position'],
                                                   mask=data['agent']['visible_mask'],
                                                   num_historical_steps=self.num_historical_steps,
                                                   num_future_steps=self.num_future_steps)
        target_traj = target_traj[agent_mask]
        target_mask = target_mask[agent_mask]
        agent_batch = data['agent']['batch'][agent_mask]
        batch_size = len(data['case_id'])

        visible_mask = data['agent']['visible_mask'][agent_mask]                      #[(n1,...nb),H+F]
        visible_num = visible_mask.sum(dim=-1)                                        #[(n1,...nb)]
        scored_mask = visible_num == self.num_historical_steps + self.num_future_steps
        scored_predict_traj = unbatch(traj_output[scored_mask,-1], agent_batch[scored_mask])                   #[(n1,K,F,2),...,(nb,K,F,2)]
        scored_target_traj = unbatch(target_traj[scored_mask,-1], agent_batch[scored_mask])                    #[(n1,F,2),...,(nb,F,2)]
        scored_target_mask = unbatch(target_mask[scored_mask,-1], agent_batch[scored_mask])                    #[(n1,F),...,(nb,F)]

        self.min_joint_ade.update(scored_predict_traj, scored_target_traj, scored_target_mask)
        self.min_joint_fde.update(scored_predict_traj, scored_target_traj, scored_target_mask)
        self.log('val_full_minJointADE', self.min_joint_ade, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log('val_full_minJointFDE', self.min_joint_fde, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)

    def _smooth_with_future_tail(self, data: Batch, eval_result: dict) -> torch.Tensor:
        best_pred = eval_result['best_joint_pred']
        if best_pred.numel() == 0 or 'future_tail_mask' not in data['agent']:
            return best_pred

        agent_mask = eval_result['agent_mask']
        future_tail_mask = self._slice_future(data['agent']['future_tail_mask'])[agent_mask]
        model_input_mask = self._agent_timeline_mask(data, 'model_input_mask')[agent_mask]
        visible_tail_mask = future_tail_mask & self._slice_future(model_input_mask)
        tail_target = eval_result['target_traj']
        smoothed = best_pred.clone()

        anchors = []
        anchor_agents = []
        for agent_idx in range(best_pred.size(0)):
            tail_indices = torch.nonzero(visible_tail_mask[agent_idx], as_tuple=False).flatten()
            if tail_indices.numel() == 0:
                continue
            tail_indices = tail_indices[-3:]
            points = tail_target[agent_idx, tail_indices]
            if points.size(0) < 3:
                points = torch.cat([points[:1].repeat(3 - points.size(0), 1), points], dim=0)
            anchors.append(points)
            anchor_agents.append(agent_idx)

        if not anchors:
            return smoothed

        anchor_agents = torch.tensor(anchor_agents, dtype=torch.long, device=best_pred.device)
        anchor_points = torch.stack(anchors, dim=0)
        smoothed[anchor_agents] = sigmoid_smooth_social_force(best_pred[anchor_agents], anchor_points)
        return smoothed

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM,
                                    nn.LSTMCell, nn.GRU, nn.GRUCell)
        blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name
                if 'bias' in param_name:
                    no_decay.add(full_param_name)
                elif 'weight' in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ('weight' in param_name or 'bias' in param_name):
                    no_decay.add(full_param_name)
        param_dict = {param_name: param for param_name, param in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {"params": [param_dict[param_name] for param_name in sorted(list(decay))],
             "weight_decay": self.weight_decay},
            {"params": [param_dict[param_name] for param_name in sorted(list(no_decay))],
             "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=self.lr, weight_decay=self.weight_decay)
        
        warmup_epochs = self.warmup_epochs
        T_max = self.T_max

        def warmup_cosine_annealing_schedule(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            return 0.5 * (1.0 + math.cos(math.pi * (epoch - warmup_epochs + 1) / (T_max - warmup_epochs + 1)))

        scheduler = {
            'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine_annealing_schedule),
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [scheduler]
