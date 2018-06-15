# -*- coding:utf-8 -*-
from pandas._libs.parsers import k

from sklearn_like_toolkit.ClassifierPack import ClassifierPack
from data_handler.DatasetLoader import DatasetLoader
from sklearn_like_toolkit.FoldingHardVote import FoldingHardVote
from sklearn_like_toolkit.warpper.mlxtend_wrapper import mlxStackingCVClf, mlxStackingClf
from util.Logger import pprint_logger, Logger
import numpy as np
import pandas as pd
from tqdm import trange

from util.deco import deco_timeit, deco_save_log
from util.misc_util import path_join

########################################################################################################################
# print(built-in function) is not good for logging


bprint = print
logger = Logger('bench_code', level='INFO')
print = logger.info
pprint = pprint_logger(print)


#######################################################################################################################


def onehot_label_smooth(Ys, smooth=0.05):
    Ys *= (1 - smooth * 2)
    Ys += smooth
    return Ys


def finger_print(size, head='_'):
    ret = head
    h_list = [c for c in '01234556789qwertyuiopasdfghjklzxcvbnm']
    for i in range(size):
        ret += np.random.choice(h_list)
    return ret


def main():
    # exp_titanic_statistic()
    # print(exp_titanic_statistic.__name__)
    # exp_stackingCV_metaclf()
    # exp_stacking_metaclf()
    # exp_voting()
    pass
