from src.Controllers.ControllerHyper import Controller as HyperController
from src.Controllers.ControllerBase import Controller as BaseController

from src.modules.SimpleHyper import HyperNN
from src.modules.MinigridPolicy import PolicyNN
from src.modules.CopyNTM import CopyNTM as NTM

# TMP minigrid
MAX_SIZE = 32 * 64 * 2 * 2
Z_NUM = 4


def builder_hyper():
    return HyperController(PolicyNN(), HyperNN(
        z_dim=32,
        z_num=Z_NUM,
        out_features=MAX_SIZE,
        z_v_evolve_prob=0.5
    ))


def builder_base():
    return BaseController(PolicyNN())

