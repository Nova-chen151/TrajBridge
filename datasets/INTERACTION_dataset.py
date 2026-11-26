from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from itertools import permutations
from itertools import product

import numpy as np
try:
    import pandas as pd
except ImportError:
    pd = None
import torch
from torch_geometric.data import Dataset
from torch_geometric.data import HeteroData
from tqdm import tqdm

try:
    import lanelet2
    from lanelet2.projection import UtmProjector
except ImportError:  # Keep module importable when only processed data is inspected.
    lanelet2 = None
    UtmProjector = None

from datasets.reconstruction_masks import add_reconstruction_masks
from utils import compute_angles_lengths_2D
from utils import transform_point_to_local_coordinate
from utils import get_index_of_A_in_B
from utils import wrap_angle


class INTERACTIONDataset(Dataset):
    def __init__(self,
                 root: str,
                 split: str,
                 transform: Optional[Callable] = None,
                 num_historical_steps: int = 10,
                 num_future_steps:int = 30,
                 task_mode: str = 'reconstruction',
                 reconstruction_tail_lens: Optional[List[int]] = None,
                 tail_len: Optional[int] = None,
                 mixed_tail_len: bool = True,
                 gap_len: Optional[int] = None,
                 window_protocol: str = 'fixed_total',
                 allow_processed_only: bool = True,
                 processed_only: bool = False,
                 max_raw_files: Optional[int] = None,
                 max_cases_per_file: Optional[int] = None,
                 scenario_filter: Optional[List[str]] = None,
                 case_id_filter: Optional[List[int]] = None,
                 processed_dir_suffix: str = '') -> None:
        self.root = root

        if split == 'train':
            self._directory = 'train'
        elif split == 'val':
            self._directory = 'val'
        elif split == 'test':
            self._directory = 'test_multi-agent'
        else:
            raise ValueError(split + ' is not valid')

        self.allow_processed_only = allow_processed_only
        self.processed_only = processed_only
        self.max_raw_files = max_raw_files
        self.max_cases_per_file = max_cases_per_file
        self.scenario_filter = set(scenario_filter or [])
        self.case_id_filter = set(int(case_id) for case_id in case_id_filter) if case_id_filter else None
        self.processed_dir_suffix = processed_dir_suffix
        self._raw_file_names = self._discover_raw_file_names()
        self._processed_file_names = self._build_processed_file_names()
        self._processed_paths = [os.path.join(self.processed_dir, name) for name in self.processed_file_names]
        
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.num_steps = num_historical_steps + num_future_steps
        self.task_mode = task_mode
        self.reconstruction_tail_lens = reconstruction_tail_lens or [0, 5, 10, 15]
        self.tail_len = tail_len
        self.mixed_tail_len = mixed_tail_len
        self.gap_len = gap_len
        self.window_protocol = window_protocol

        if lanelet2 is not None:
            self.projector = UtmProjector(lanelet2.io.Origin(0, 0))
            self.traffic_rules = lanelet2.traffic_rules.create(lanelet2.traffic_rules.Locations.Germany,
                                                               lanelet2.traffic_rules.Participants.Vehicle)
        else:
            self.projector = None
            self.traffic_rules = None

        self._agent_type = ['car', 'pedestrian/bicycle']
        self._polyline_side = ['left','center','right']
        super(INTERACTIONDataset, self).__init__(root=root, transform=transform)

    def _discover_raw_file_names(self) -> List[str]:
        if not os.path.isdir(self.raw_dir):
            return []
        raw_file_names = sorted(name for name in os.listdir(self.raw_dir) if name.endswith('.csv'))
        if self.scenario_filter:
            raw_file_names = [
                name for name in raw_file_names
                if self._infer_scenario_name(name) in self.scenario_filter
            ]
        if self.max_raw_files is not None:
            raw_file_names = raw_file_names[:self.max_raw_files]
        return raw_file_names

    def _discover_processed_file_names(self) -> List[str]:
        if not os.path.isdir(self.processed_dir):
            return []
        metadata_files = {'pre_filter.pt', 'pre_transform.pt'}
        return sorted(
            name for name in os.listdir(self.processed_dir)
            if name.endswith('.pt') and name not in metadata_files
        )

    @staticmethod
    def _infer_scenario_name(raw_file_name: str) -> str:
        stem = os.path.splitext(raw_file_name)[0]
        for suffix in ('_train', '_val', '_test', '_test_multi-agent', '_test_conditional-multi-agent'):
            if stem.endswith(suffix):
                return stem[:-len(suffix)]
        parts = stem.split('_')
        return '_'.join(parts[:-1]) if len(parts) > 1 else stem

    @staticmethod
    def _require_column(df: pd.DataFrame, candidates: List[str], semantic_name: str) -> str:
        for column in candidates:
            if column in df.columns:
                return column
        raise KeyError(
            f'Missing required column for {semantic_name}. Tried {candidates}. '
            f'Available columns: {list(df.columns)}'
        )

    @staticmethod
    def _optional_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        for column in candidates:
            if column in df.columns:
                return column
        return None

    def _select_case_ids(self, df: pd.DataFrame) -> List[int]:
        case_col = self._require_column(df, ['case_id', 'caseId', 'case'], 'case_id')
        case_ids = [int(case_id) for case_id in df[case_col].dropna().unique()]
        if self.case_id_filter is not None:
            case_ids = [case_id for case_id in case_ids if case_id in self.case_id_filter]
        if self.max_cases_per_file is not None:
            case_ids = case_ids[:self.max_cases_per_file]
        return case_ids

    @staticmethod
    def _normalize_agent_type(value) -> str:
        value = str(value).lower()
        if value in ('car', 'vehicle', 'motor_vehicle'):
            return 'car'
        if value in ('pedestrian/bicycle', 'pedestrian', 'bicycle', 'cyclist'):
            return 'pedestrian/bicycle'
        raise ValueError(f'Unsupported agent_type: {value}')

    def _build_processed_file_names(self) -> List[str]:
        processed_file_names = []
        processed_candidates = self._discover_processed_file_names()

        if self.processed_only:
            if processed_candidates:
                return processed_candidates
            raise FileNotFoundError(
                'processed_only=True but no processed .pt files were found in the *_processed folder.'
            )

        if lanelet2 is None and self.allow_processed_only and processed_candidates:
            return processed_candidates

        if self._raw_file_names and pd is not None:
            for raw_path in tqdm(self.raw_paths):
                raw_dir, raw_file_name = os.path.split(raw_path)
                scenario_name = self._infer_scenario_name(raw_file_name)
                df = pd.read_csv(raw_path)
                for case_id in tqdm(self._select_case_ids(df)):
                    processed_file_names.append(scenario_name + '_' + str(int(case_id)) + '.pt')
            return processed_file_names

        if self.allow_processed_only and processed_candidates:
            return processed_candidates

        if self._raw_file_names and pd is None:
            raise ImportError(
                'pandas is required to infer INTERACTION processed filenames from raw CSV files. '
                'Install pandas in the TrajBridge environment or provide preprocessed .pt files.'
            )

        raise FileNotFoundError(
            'No raw files or processed .pt files found. Please provide INTERACTION root with '
            'train/val/test_multi-agent and maps directories, or preprocessed *_processed folders.'
        )

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self._directory)

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self._directory + '_processed' + self.processed_dir_suffix)
    
    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return self._raw_file_names

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return self._processed_file_names

    @property
    def processed_paths(self) -> List[str]:
        return self._processed_paths

    def process(self) -> None:
        if pd is None:
            raise ImportError('pandas is required to process raw INTERACTION CSV files.')
        if lanelet2 is None:
            raise ImportError('lanelet2 is required to process raw INTERACTION maps, but it is not installed.')

        for raw_path in tqdm(self.raw_paths):
            #map
            raw_dir, raw_file_name = os.path.split(raw_path)
            scenario_name = self._infer_scenario_name(raw_file_name)
            base_dir = os.path.dirname(raw_dir)
            map_path = os.path.join(base_dir,'maps',scenario_name+'.osm')
            print(f'raw_file_name: {raw_file_name}')
            print(f'inferred scenario_name: {scenario_name}')
            print(f'map_path: {map_path}')
            print(f'map exists: {os.path.exists(map_path)}')
            if not os.path.exists(map_path):
                raise FileNotFoundError(f'Map file not found: {map_path}')
            map_api = lanelet2.io.load(map_path, self.projector)
            routing_graph = lanelet2.routing.RoutingGraph(map_api, self.traffic_rules)
            
            #agent
            df = pd.read_csv(raw_path)
            case_col = self._require_column(df, ['case_id', 'caseId', 'case'], 'case_id')
            selected_case_ids = self._select_case_ids(df)
            for case_id in tqdm(selected_case_ids):
                case_df = df[df[case_col].astype(int) == int(case_id)]
                data = dict()      
                data['scenario_name'] = scenario_name
                data['case_id'] = int(case_id)
                data.update(self.get_features(case_df, map_api, routing_graph))
                torch.save(data, os.path.join(self.processed_dir, scenario_name + '_' + str(int(case_id)) + '.pt'))

    def get_features(self, 
                     df: pd.DataFrame,
                     map_api,
                     routing_graph) -> Dict:
        data = {
            'agent': {},
            'lane': {},
            'polyline': {},
            ('polyline', 'lane'): {},
            ('lane', 'lane'): {}
        }
        #AGENT
        # filter out actors that are unseen during the historical time steps
        case_col = self._require_column(df, ['case_id', 'caseId', 'case'], 'case_id')
        track_col = self._require_column(df, ['track_id', 'trackId', 'track'], 'track_id')
        timestamp_col = self._require_column(df, ['timestamp_ms', 'timestamp', 'time_ms'], 'timestamp_ms')
        x_col = self._require_column(df, ['x', 'x_m'], 'x')
        y_col = self._require_column(df, ['y', 'y_m'], 'y')
        vx_col = self._require_column(df, ['vx', 'vx_mps'], 'vx')
        vy_col = self._require_column(df, ['vy', 'vy_mps'], 'vy')
        agent_type_col = self._require_column(df, ['agent_type', 'type', 'object_type'], 'agent_type')
        heading_col = self._optional_column(df, ['psi_rad', 'heading', 'yaw_rad'])
        length_col = self._optional_column(df, ['length', 'length_m'])
        width_col = self._optional_column(df, ['width', 'width_m'])

        timestep_ids = list(np.sort(df[timestamp_col].unique()))
        historical_timestamps = timestep_ids[:self.num_historical_steps]
        historical_df = df[df[timestamp_col].isin(historical_timestamps)]
        agent_ids = list(historical_df[track_col].unique())
        num_agents = len(agent_ids)
        df = df[df[track_col].isin(agent_ids)]

        # initialization
        agent_id = torch.zeros(num_agents, dtype=torch.int64)
        visible_mask = torch.zeros(num_agents, self.num_steps, dtype=torch.bool)
        agent_position = torch.zeros(num_agents, self.num_steps, 2, dtype=torch.float)
        agent_heading = torch.zeros(num_agents, self.num_steps, dtype=torch.float)
        agent_velocity = torch.zeros(num_agents, self.num_steps, 2, dtype=torch.float)
        agent_velocity_length = torch.zeros(num_agents, self.num_historical_steps, dtype=torch.float)
        agent_velocity_theta = torch.zeros(num_agents, self.num_historical_steps, dtype=torch.float)
        agent_length = torch.zeros(num_agents, dtype=torch.float)
        agent_width = torch.zeros(num_agents, dtype=torch.float)
        agent_type = torch.zeros(num_agents, dtype=torch.uint8)
        agent_category = torch.zeros(num_agents, dtype=torch.uint8)
        agent_interset = torch.zeros(num_agents, dtype=torch.uint8)
        
        for track_id, track_df in df.groupby(track_col):
            track_df = track_df.sort_values(timestamp_col)
            agent_idx = agent_ids.index(track_id)
            agent_id[agent_idx] = int(track_id)
            agent_steps_all = [timestep_ids.index(timestamp) for timestamp in track_df[timestamp_col]]
            valid_rows = [step < self.num_steps for step in agent_steps_all]
            if not any(valid_rows):
                continue
            agent_steps = [step for step, keep in zip(agent_steps_all, valid_rows) if keep]
            track_df = track_df.iloc[np.array(valid_rows, dtype=bool)]

            visible_mask[agent_idx, agent_steps] = True

            agent_type_name = self._normalize_agent_type(track_df[agent_type_col].values[0])
            agent_type[agent_idx] = torch.tensor(self._agent_type.index(agent_type_name), dtype=torch.uint8)

            if agent_type_name == 'car':
                if length_col is not None:
                    agent_length[agent_idx] = track_df[length_col].values[0]
                if width_col is not None:
                    agent_width[agent_idx] = track_df[width_col].values[0]

            if 'track_to_predict' in track_df.columns:
                agent_category[agent_idx] = torch.tensor(track_df['track_to_predict'].values[0], dtype=torch.uint8)
                agent_interset[agent_idx] = torch.tensor(track_df['interesting_agent'].values[0], dtype=torch.uint8)
            elif agent_type_name == 'car':
                agent_category[agent_idx] = torch.tensor(1, dtype=torch.uint8)


            agent_position[agent_idx, agent_steps] = torch.from_numpy(np.stack([track_df[x_col].values, track_df[y_col].values], axis=-1)).float()

            agent_velocity[agent_idx, agent_steps] = torch.from_numpy(np.stack([track_df[vx_col].values, track_df[vy_col].values], axis=-1)).float()
            velocity_length, velocity_theta = compute_angles_lengths_2D(agent_velocity[agent_idx])
            agent_velocity_length[agent_idx] = velocity_length[:self.num_historical_steps]

            if agent_type_name == 'car' and heading_col is not None:
                agent_heading[agent_idx,agent_steps] = torch.from_numpy(track_df[heading_col].values).float()
                agent_velocity_theta[agent_idx] = wrap_angle(velocity_theta[:self.num_historical_steps] - agent_heading[agent_idx,:self.num_historical_steps])
            else:
                agent_heading[agent_idx] = velocity_theta
                agent_velocity_theta[agent_idx] = 0


        data['agent']['id'] = agent_id
        data['agent']['num_nodes'] = num_agents
        data['agent']['visible_mask'] = visible_mask
        data['agent']['position'] = agent_position
        data['agent']['heading'] = agent_heading
        data['agent']['velocity_length'] = agent_velocity_length
        data['agent']['velocity_theta'] = agent_velocity_theta
        data['agent']['length'] = agent_length
        data['agent']['width'] = agent_width
        data['agent']['type'] = agent_type
        data['agent']['category'] = agent_category
        data['agent']['interest'] = agent_interset
        
        #MAP
        valid_lanes = []
        for lane in map_api.laneletLayer:
            if len(lane.centerline) < 2 or len(lane.leftBound) < 2 or len(lane.rightBound) < 2:
                print(f'Warning: skipped lane {lane.id} because it has fewer than 2 boundary/centerline points.')
                continue
            valid_lanes.append(lane)

        if len(valid_lanes) == 0:
            raise ValueError('No valid lanes found in map; MapEncoder cannot run without lane nodes.')

        lane_ids = []
        for lane in valid_lanes:
            lane_ids.append(lane.id)
        
        num_lanes = len(lane_ids)
        lane_id = torch.zeros(num_lanes, dtype=torch.float)
        lane_position = torch.zeros(num_lanes, 2, dtype=torch.float)
        lane_heading = torch.zeros(num_lanes, dtype=torch.float)
        lane_length = torch.zeros(num_lanes, dtype=torch.float)

        num_polylines = torch.zeros(num_lanes, dtype=torch.long)
        polyline_position: List[Optional[torch.Tensor]] = [None] * num_lanes
        polyline_heading: List[Optional[torch.Tensor]] = [None] * num_lanes
        polyline_length: List[Optional[torch.Tensor]] = [None] * num_lanes
        polyline_side: List[Optional[torch.Tensor]] = [None] * num_lanes

        lane_left_neighbor_edge_index = []
        lane_right_neighbor_edge_index = []
        lane_predecessor_edge_index = []
        lane_successor_edge_index = []

        for lane in valid_lanes:
            lane_idx = lane_ids.index(lane.id)
            lane_id[lane_idx] = lane.id

            points = [np.array([pt.x, pt.y]) for pt in lane.centerline]
            centerline = torch.from_numpy(np.array(points)).float()

            center_index = int((centerline.size(0)-1)/2)
            lane_position[lane_idx] = centerline[center_index, :2]
            lane_heading[lane_idx] = torch.atan2(centerline[center_index+1, 1] - centerline[center_index, 1], 
                                                 centerline[center_index+1, 0] - centerline[center_index, 0])
            lane_length[lane_idx] = torch.norm(centerline[1:] - centerline[:-1], p=2, dim=-1).sum()
            
            points = [np.array([pt.x, pt.y]) for pt in lane.leftBound]
            left_boundary = torch.from_numpy(np.array(points)).float()
            points = [np.array([pt.x, pt.y]) for pt in lane.rightBound]
            right_boundary = torch.from_numpy(np.array(points)).float()
            left_vector = left_boundary[1:] - left_boundary[:-1]      
            right_vector = right_boundary[1:] - right_boundary[:-1]
            centerline_vector = centerline[1:] - centerline[:-1]
            polyline_position[lane_idx] = torch.cat([(left_boundary[1:] + left_boundary[:-1])/2, (right_boundary[1:] + right_boundary[:-1])/2, (centerline[1:] + centerline[:-1])/2], dim=0)
            polyline_length[lane_idx], polyline_heading[lane_idx] = compute_angles_lengths_2D(torch.cat([left_vector, right_vector, centerline_vector],dim=0))
            num_left_polyline = len(left_vector)
            num_right_polyline = len(right_vector)
            num_centerline_polyline = len(centerline_vector)
            polyline_side[lane_idx] = torch.cat(
                [torch.full((num_left_polyline,), self._polyline_side.index('left'), dtype=torch.uint8),
                 torch.full((num_right_polyline,), self._polyline_side.index('right'), dtype=torch.uint8),
                 torch.full((num_centerline_polyline,), self._polyline_side.index('center'), dtype=torch.uint8)], dim=0)

            num_polylines[lane_idx] = num_left_polyline + num_right_polyline + num_centerline_polyline

            lane_left_neighbor_lane = routing_graph.left(lane)
            lane_left_neighbor_id = [lane_left_neighbor_lane.id] if lane_left_neighbor_lane else []
            lane_left_neighbor_idx = get_index_of_A_in_B(lane_left_neighbor_id, lane_ids)
            if len(lane_left_neighbor_idx) != 0:
                edge_index = torch.stack([torch.tensor(lane_left_neighbor_idx, dtype=torch.long), torch.full((len(lane_left_neighbor_idx),), lane_idx, dtype=torch.long)], dim=0)
                lane_left_neighbor_edge_index.append(edge_index)
            lane_right_neighbor_lane = routing_graph.right(lane)
            lane_right_neighbor_id = [lane_right_neighbor_lane.id] if lane_right_neighbor_lane else []
            lane_right_neighbor_idx = get_index_of_A_in_B(lane_right_neighbor_id, lane_ids)
            if len(lane_right_neighbor_idx) != 0:
                edge_index = torch.stack([torch.tensor(lane_right_neighbor_idx, dtype=torch.long), torch.full((len(lane_right_neighbor_idx),), lane_idx, dtype=torch.long)], dim=0)
                lane_right_neighbor_edge_index.append(edge_index)
            lane_predecessor_lanes = routing_graph.previous(lane)
            lane_predecessor_ids = [ll.id for ll in lane_predecessor_lanes] if lane_predecessor_lanes else []
            lane_predecessor_idx = get_index_of_A_in_B(lane_predecessor_ids, lane_ids)
            if len(lane_predecessor_idx) != 0:
                edge_index = torch.stack([torch.tensor(lane_predecessor_idx, dtype=torch.long), torch.full((len(lane_predecessor_idx),), lane_idx, dtype=torch.long)], dim=0)
                lane_predecessor_edge_index.append(edge_index)
            lane_successor_lanes = routing_graph.following(lane)
            lane_successor_ids = [ll.id for ll in lane_successor_lanes] if lane_successor_lanes else []
            lane_successor_idx = get_index_of_A_in_B(lane_successor_ids, lane_ids)
            if len(lane_successor_idx) != 0:
                edge_index = torch.stack([torch.tensor(lane_successor_idx, dtype=torch.long), torch.full((len(lane_successor_idx),), lane_idx, dtype=torch.long)], dim=0)
                lane_successor_edge_index.append(edge_index)

        data['lane']['id'] = lane_id
        data['lane']['num_nodes'] = num_lanes
        data['lane']['position'] = lane_position
        data['lane']['length'] = lane_length
        data['lane']['heading'] = lane_heading
        data['lane']['visible_mask'] = torch.ones(num_lanes, dtype=torch.bool)

        data['polyline']['num_nodes'] = num_polylines.sum().item()
        if data['polyline']['num_nodes'] == 0:
            raise ValueError('No valid polylines found in map; MapEncoder cannot run without polyline nodes.')
        data['polyline']['position'] = torch.cat(polyline_position, dim=0)
        data['polyline']['heading'] = torch.cat(polyline_heading, dim=0)
        data['polyline']['length'] = torch.cat(polyline_length, dim=0)
        data['polyline']['side'] = torch.cat(polyline_side, dim=0)

        polyline_to_lane_edge_index = torch.stack([torch.arange(num_polylines.sum(), dtype=torch.long), torch.arange(num_lanes, dtype=torch.long).repeat_interleave(num_polylines)], dim=0)
        data['polyline', 'lane']['polyline_to_lane_edge_index'] = polyline_to_lane_edge_index

        if len(lane_left_neighbor_edge_index) != 0:
            lane_left_neighbor_edge_index = torch.cat(lane_left_neighbor_edge_index, dim=1)
        else:
            lane_left_neighbor_edge_index = torch.tensor([[], []], dtype=torch.long)
        if len(lane_right_neighbor_edge_index) != 0:
            lane_right_neighbor_edge_index = torch.cat(lane_right_neighbor_edge_index, dim=1)
        else:
            lane_right_neighbor_edge_index = torch.tensor([[], []], dtype=torch.long)
        if len(lane_predecessor_edge_index) != 0:
            lane_predecessor_edge_index = torch.cat(lane_predecessor_edge_index, dim=1)
        else:
            lane_predecessor_edge_index = torch.tensor([[], []], dtype=torch.long)
        if len(lane_successor_edge_index) != 0:
            lane_successor_edge_index = torch.cat(lane_successor_edge_index, dim=1)
        else:
            lane_successor_edge_index = torch.tensor([[], []], dtype=torch.long)
        
        data['lane', 'lane']['left_neighbor_edge_index'] = lane_left_neighbor_edge_index
        data['lane', 'lane']['right_neighbor_edge_index'] = lane_right_neighbor_edge_index
        data['lane', 'lane']['predecessor_edge_index'] = lane_predecessor_edge_index
        data['lane', 'lane']['successor_edge_index'] = lane_successor_edge_index

        return data

    def len(self) -> int:
        return len(self.processed_file_names)

    def get(self, idx: int) -> HeteroData:
        data = HeteroData(torch.load(self.processed_paths[idx]))
        if 'lane' in data and 'visible_mask' not in data['lane']:
            data['lane']['visible_mask'] = torch.ones(int(data['lane']['num_nodes']), dtype=torch.bool)
        return add_reconstruction_masks(
            data=data,
            history_len=self.num_historical_steps,
            future_len=self.num_future_steps,
            task_mode=self.task_mode,
            reconstruction_tail_lens=self.reconstruction_tail_lens,
            tail_len=self.tail_len,
            mixed_tail_len=self.mixed_tail_len,
            gap_len=self.gap_len,
            window_protocol=self.window_protocol,
        )
