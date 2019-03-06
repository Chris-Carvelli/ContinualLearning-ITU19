from src.models.EvolvableModelHyper import EvolvableModel as HyperModel
from src.models.EvolvableModelBase import EvolvableModel as BaseModel
from src.models.EvolvableModelNTM import EvolvableModel as NTMModel

from src.modules.SimpleHyper import HyperNN
from src.modules.MinigridPolicy import PolicyNN
from src.modules.CopyNTM import CopyNTM as NTM

# TMP minigrid
MAX_SIZE = 32 * 64 * 2 * 2
Z_NUM = 4


def builder_hyper():
    return HyperModel(PolicyNN(), HyperNN(
        z_dim=32,
        z_num=Z_NUM,
        out_features=MAX_SIZE,
        z_v_evolve_prob=0.5
    ))


def builder_base():
    return BaseModel(PolicyNN())


def builder_ntm(copy_size):
    return NTMModel(NTM(
        copy_size=copy_size))
