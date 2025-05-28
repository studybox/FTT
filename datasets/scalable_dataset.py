import os
import os.path as osp
import pickle
from typing import Callable, List, Optional, Tuple, Union
import pandas as pd
import torch 
from torch_geometric.data import Dataset
# from smart.utils.log import Logging
import numpy as np
from .preprocess import TokenProcessor
from datasets.scenario_dataset import ScenarioData
from models.smart.transforms import WaymoTargetBuilder

def distance(point1, point2):
    return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)


class MultiDataset(Dataset):
    def __init__(self, config, processed_files=None, transform=None, pre_transform=None):
        self.config = config
        if isinstance(config["data_dir"], str):
            self.root = root = config["data_dir"]
            self.raw_files = [f for f in os.listdir(osp.join(root, "raw")) if 'dump' in f]
            self.map_files = [f for f in os.listdir(osp.join(root, "raw")) if 'lanelet' in f]
        else:
            self.root = root = config["data_dir"][0]
            self.raw_files = [f for f in os.listdir(osp.join(root, "raw")) if 'dump' in f]
            self.map_files = [f for f in os.listdir(osp.join(root, "raw")) if 'lanelet' in f]
        if processed_files is None:
            raise ValueError("processed_files must be provided")
        else:
            self._processed_file_names = processed_files
            self._num_samples = len(processed_files)
        num_historical_steps = 11
        num_future_steps = 30
        transform = WaymoTargetBuilder(num_historical_steps, num_future_steps)

        # self.well_done = [0]
        # if split not in ('train', 'val', 'test'):
        #     raise ValueError(f'{split} is not a valid split')
        # self.split = split
        # self.training = split == 'train'
        # # self.logger.debug("Starting loading dataset")
        # self._raw_file_names = []
        # self._raw_paths = []
        # self._raw_file_dataset = []
        # if raw_dir is not None:
        #     self._raw_dir = raw_dir
        #     for raw_dir in self._raw_dir:
        #         raw_dir = os.path.expanduser(os.path.normpath(raw_dir))
        #         dataset = "waymo"
        #         file_list = os.listdir(raw_dir)
        #         self._raw_file_names.extend(file_list)
        #         self._raw_paths.extend([os.path.join(raw_dir, f) for f in file_list])
        #         self._raw_file_dataset.extend([dataset for _ in range(len(file_list))])
        # if self.root is not None:
        #     split_datainfo = os.path.join(root, "split_datainfo.pkl")
        #     with open(split_datainfo, 'rb+') as f:
        #         split_datainfo = pickle.load(f)
        #     if split == "test":
        #         split = "val"
        #     self._processed_file_names = split_datainfo[split]
        # self.dim = dim
        # self.num_historical_steps = num_historical_steps
        # self._num_samples = len(self._processed_file_names) - 1 if processed_dir is not None else len(self._raw_file_names)
        # self.logger.debug("The number of {} dataset is ".format(split) + str(self._num_samples))
        self.token_processor = TokenProcessor(2048)

        super(MultiDataset, self).__init__(root=root, transform=transform, pre_transform=None, pre_filter=None)

    # @property
    # def raw_dir(self) -> str:
    #     return self._raw_dir

    # @property
    # def raw_paths(self) -> List[str]:
    #     return self._raw_paths

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return self.raw_files

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return self._processed_file_names

    def len(self) -> int:
        return self._num_samples

    def generate_ref_token(self):
        pass

    def get(self, idx: int):
        # with open(self.raw_paths[idx], 'rb') as handle:
        #     data = pickle.load(handle)
        f = self.processed_file_names[idx]
        old_data = torch.load(f)
        kwargs = {k: v for k, v in old_data.__dict__.items() if not k.startswith('_')}
        # Reconstruct with new ScenarioData class binding
        data = ScenarioData(**kwargs)
        data = self.token_processor.preprocess(data)
        return data
    
