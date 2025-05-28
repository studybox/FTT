from .interaction_dataset import InteractionDataset
from .scenario_dataset import ScenarioDataset
from .scalable_dataset import MultiDataset

from .collate_functions import traj_collate, vector_collate, lanegcn_collate_fn
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.loader import DataLoader as GeoDataLoader
from .utils import mix_dataset

Datasets = {"ftt": ScenarioDataset, 
            "lanegcn": ScenarioDataset,
            "tnt": ScenarioDataset,
            "mfp": ScenarioDataset,
            "smart": MultiDataset,
            "fjmp": InteractionDataset}

collate_fns = {"ftt": traj_collate,
                "lanegcn": traj_collate,
                "tnt": vector_collate,
                "mfp": traj_collate,
               "smart": None, 
               "fjmp": lanegcn_collate_fn}

Dataloaders = {"ftt": TorchDataLoader, 
               "lanegcn": TorchDataLoader,
                "tnt": TorchDataLoader, 
                "mfp": TorchDataLoader,
               "smart": GeoDataLoader, 
               "fjmp": TorchDataLoader}
