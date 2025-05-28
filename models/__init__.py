from .ftt.ftt import FTT
from .smart.smart import SMART
from .fjmp.fjmp import FJMP
from .laneGCN.laneGCN import LaneGCN
from .tnt.tnt import TNT
from .mfp.mfp import MFP

Models = {"ftt": FTT, "smart": SMART, "fjmp": FJMP, "lanegcn": LaneGCN, "tnt": TNT, "mfp": MFP}