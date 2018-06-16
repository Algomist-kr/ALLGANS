# -*- coding:utf-8 -*-


########################################################################################################################
# print(built-in function) is not good for logging
from sklearn_like_toolkit.ClassifierPack import ClassifierPack

from data_handler.DatasetPackLoader import DatasetPackLoader
from sklearn_like_toolkit.FoldingHardVote import FoldingHardVote
from sklearn_like_toolkit.warpper.mlxtend_wrapper import mlxStackingCVClf, mlxStackingClf
from util.Logger import StdoutOnlyLogger, pprint_logger
import numpy as np
import pandas as pd
from tqdm import trange

from util.deco import deco_timeit, deco_save_log
from util.misc_util import path_join

bprint = print
logger = StdoutOnlyLogger()
print = logger.get_log()
pprint = pprint_logger(print)
#######################################################################################################################
@deco_timeit
@deco_save_log
def exp_stacking_metaclf(print, pprint):
    dataset = DatasetPackLoader().load_dataset("titanic")
    train_set, valid_set = dataset.split('train', 'train', 'valid', (7, 3))
    train_Xs, train_Ys = train_set.full_batch(['Xs', 'Ys'])
    valid_Xs, valid_Ys = valid_set.full_batch(['Xs', 'Ys'])

    clf = ClassifierPack()
    clf.drop_clf('mlxMLP')
    clf.drop_clf('mlxAdaline')
    clf.drop_clf('mlxSoftmaxRegressionClf')
    clf.drop_clf('skGaussian_NB')
    clf.drop_clf('skQDA')

    pack = clf.pack
    for key, meta_clf in pack.items():
        if 'mlx' in key:
            continue

        pprint(f'meta clf = {key}')
        stacking = clf.make_stackingClf(meta_clf)
        stacking.fit(train_Xs, train_Ys)
        score = stacking.score(valid_Xs, valid_Ys)
        pprint(f'score {score}')
        # break


@deco_timeit
@deco_save_log
def exp_stackingCV_metaclf(print, pprint):
    dataset = DatasetPackLoader().load_dataset("titanic")
    train_set, valid_set = dataset.split('train', 'train', 'valid', (7, 3))
    train_Xs, train_Ys = train_set.full_batch(['Xs', 'Ys'])
    valid_Xs, valid_Ys = valid_set.full_batch(['Xs', 'Ys'])

    clf = ClassifierPack()
    clf.drop_clf('mlxMLP')
    clf.drop_clf('mlxAdaline')
    clf.drop_clf('mlxSoftmaxRegressionClf')
    clf.drop_clf('skGaussian_NB')
    clf.drop_clf('skQDA')

    pack = clf.pack
    for key, meta_clf in pack.items():
        if 'mlx' in key:
            continue

        pprint(f'meta clf = {key}')
        stacking = clf.make_stackingCVClf(meta_clf)
        stacking.fit(train_Xs, train_Ys)
        score = stacking.score_pack(valid_Xs, valid_Ys)
        pprint(f'score {score}')
        # break

@deco_timeit
@deco_save_log
def exp_titanic_statistic(print, pprint):
    dataset = DatasetPackLoader().load_dataset("titanic")
    train_set, valid_set = dataset.split('train', 'train', 'valid', (7, 3))
    train_Xs, train_Ys = train_set.full_batch(['Xs', 'Ys'])
    valid_Xs, valid_Ys = valid_set.full_batch(['Xs', 'Ys'])

    clf_pack = ClassifierPack()
    clf_pack.drop_clf('mlxMLP')
    clf_pack.drop_clf('mlxAdaline')
    clf_pack.drop_clf('mlxSoftmaxRegressionClf')
    clf_pack.drop_clf('skGaussian_NB')
    clf_pack.drop_clf('skQDA')

    pack = clf_pack.pack
    pprint(f'pack list {pack}')
    meta_clf = pack['skBernoulli_NB']
    pprint(f'metaclf = {meta_clf}')

    clf_pack.fit(train_Xs, train_Ys)
    score_pack = clf_pack.score_pack(valid_Xs, valid_Ys)
    pprint('default param clf pack')
    pprint(score_pack)

    clf_pack.param_search(train_Xs, train_Ys)
    score_pack = clf_pack.score_pack(valid_Xs, valid_Ys)
    pprint('optimize param clf pack top1')
    pprint(score_pack)

    pack = [clf for k, clf in clf_pack.pack.items() if hasattr(clf, 'get_params')]
    pack1_default = pack
    pack10_default = pack * 10
    pack100_default_ = pack * 100

    pack1_top1 = clf_pack.clone_top_k_tuned(k=1)
    pack1_top1 = [clf for k, clf in pack1_top1.items() if hasattr(clf, 'get_params')]
    pack10_top1 = pack1_top1 * 10
    pack100_top1 = pack1_top1 * 100

    pack1_top5 = clf_pack.clone_top_k_tuned(k=5)
    pack1_top5 = [clf for k, clf in pack1_top5.items() if hasattr(clf, 'get_params')]
    pack10_top5 = pack1_top5 * 10
    pack100_top5 = pack1_top5 * 100

    def voting_stacking_stackingCV(pack, param_type, pack_n, top):
        pprint(f'param_type={param_type}, pack_n={pack_n}, top={top}')

        voting = FoldingHardVote(pack)
        voting.fit(train_Xs, train_Ys)
        score_pack = voting.score_pack(valid_Xs, valid_Ys)
        pprint(f'{param_type} param clf pack * {pack_n}, {top} to hard voting')
        pprint(score_pack)

        stacking = mlxStackingClf(pack, meta_clf)
        stacking.fit(train_Xs, train_Ys)
        score_pack = stacking.score_pack(valid_Xs, valid_Ys)
        pprint(f'{param_type} param clf pack * {pack_n}, {top} to stacking')
        pprint(score_pack)

        stackingCV = mlxStackingCVClf(pack, meta_clf)
        stackingCV.fit(train_Xs, train_Xs)
        score_pack = stackingCV.score_pack(valid_Xs, valid_Ys)
        pprint(f'{param_type} param clf pack * {pack_n}, {top} to stackingCV')
        pprint(score_pack)

    voting_stacking_stackingCV(pack1_default, 'default', 1, None)
    voting_stacking_stackingCV(pack10_default, 'default', 10, None)
    voting_stacking_stackingCV(pack100_default_, 'default', 100, None)
    voting_stacking_stackingCV(pack1_top1, 'optimize', 1, 'top1')
    voting_stacking_stackingCV(pack10_top1, 'optimize', 10, 'top1')
    voting_stacking_stackingCV(pack100_top1, 'optimize', 100, 'top1')
    voting_stacking_stackingCV(pack1_top5, 'optimize', 1, 'top5')
    voting_stacking_stackingCV(pack10_top5, 'optimize', 10, 'top5')
    voting_stacking_stackingCV(pack100_top5, 'optimize', 100, 'top5')


@deco_timeit
@deco_save_log
def exp_titanic_id_static(print, pprint):
    dataset = DatasetPackLoader().load_dataset("titanic")
    dataset = dataset.set['train']

    ret_dict = {}
    n = 100
    for i in range(n):
        clf_pack = ClassifierPack()
        dataset.shuffle()
        train_set, valid_set = dataset.split((7, 3))
        train_Xs, train_Ys = train_set.full_batch(['Xs', 'Ys'])
        clf_pack.fit(train_Xs, train_Ys)

        dataset.sort()
        full_Xs, full_Ys = dataset.full_batch(['Xs', 'Ys'])
        predict = clf_pack.predict(full_Xs)

        for key in predict:
            if key in ret_dict:
                ret_dict[key] += predict[key] / float(n)
            else:
                ret_dict[key] = predict[key] / float(n)

    import pandas as pd

    df = pd.DataFrame()
    for key in ret_dict:
        df[key] = ret_dict[key]
    for key in dataset.BATCH_KEYS:
        if key in ['Xs', 'Ys']:
            continue
        print(key, type(key))
        df[key] = dataset.full_batch([key])

    df.to_csv('./exp_titianic_id_result.csv', )

def main():
    print(exp_titanic_statistic.__name__)
    # exp_stacking_metaclf()
    # exp_voting()
    pass



