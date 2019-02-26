from models.EvolvableModelHyper import EvolvableModel as HyperModel
from models.EvolvableModelBase import EvolvableModel as BaseModel

from src.modules.FrostbitePolicy import PolicyNN
from src.modules.SimpleHyper import HyperNN

# TMP minigrid
# MAX_SIZE = 32 * 64 * 2 * 2
# Z_NUM = 4

# TMP Frostbite
MAX_SIZE = 4 * 4 * 64 * 512
Z_NUM = 5


def builder_hyper():
    return HyperModel(PolicyNN(), HyperNN(
        z_dim=32,
        z_num=Z_NUM,
        out_features=MAX_SIZE,
        z_v_evolve_prob=0.5
    ))


def builder_base():
    return BaseModel(PolicyNN())
