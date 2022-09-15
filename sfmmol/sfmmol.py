import itertools
import random
import re

import GPy
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

from sfmmol.utils import NBGraphChecker


class StratifiedRandomSampleGrph:
    """StratifiedRandom Sample for Graph"""

    def __init__(self, strada, ns, random_seed=0, printlevel=0):
        self.grph = None
        self.pri = None
        self.cnt = None
        self.printlevel = printlevel
        # self.strada = [re.compile(r'{}'.format(_key)) for _key in strada.keys()]
        self.strada = [re.compile(_key) for _key in strada.keys()]
        # self.strada2 = [_key for _key in strada.keys()]
        _sum = sum(strada.values())
        _acc = [item / _sum for item in list(itertools.accumulate(strada.values()))]
        print(_acc)
        self.stratify = lambda x: [i for i, v in enumerate(_acc) if x < v][0]
        self.ns = ns
        random.seed(random_seed)

    def search(self, grph, n=1, node_order=None):
        """Search"""
        res = None
        if self.grph is None:
            self.grph = grph
            self.pri = list(self.grph.nodes())
            # print(self.pri)
            random.shuffle(self.pri)
            # print(self.pri[0:10])
            # print(itemgetter(*(self.pri[0:10]))(self.ns))

            while res is None:
                self.cnt = 0
                _stratum = self.stratify(random.random())
                # print(f'strata: {_stratum}', self.strada[_stratum])
                while self.cnt < len(self.pri) and (
                    "d" in grph.nodes[self.pri[self.cnt]] or self.strada[_stratum].match(self.ns[self.pri[self.cnt]]) is None
                ):
                    # if self.cnt < 10:
                    #     print(self.cnt, self.ns[self.pri[self.cnt]])
                    self.cnt += 1

                # print(">", self.cnt, self.ns[self.pri[self.cnt]])
                if self.cnt < len(self.pri):
                    res = (-1, -1, -1, -1, self.pri[self.cnt])

        else:
            while res is None:
                self.cnt = 0
                _stratum = self.stratify(random.random())
                # print(f'strata: {_stratum}', self.strada[_stratum])
                while self.cnt < len(self.pri) and (
                    "d" in grph.nodes[self.pri[self.cnt]] or self.strada[_stratum].match(self.ns[self.pri[self.cnt]]) is None
                ):
                    # if self.cnt < 10:
                    #     print(self.cnt, self.ns[self.pri[self.cnt]])
                    self.cnt += 1

                # print(">", self.cnt, self.ns[self.pri[self.cnt]])
                if self.cnt < len(self.pri):
                    res = (-1, -1, -1, -1, self.pri[self.cnt])

        if 0 < self.printlevel:
            print("StratifiedRandomSampleGrph:", res, self.ns[self.pri[self.cnt]])

        return res


class RandomSampleGrph:
    """Random Sample for Graph"""

    def __init__(self, random_seed=0):
        self.grph = None
        self.pri = None
        self.cnt = None
        random.seed(random_seed)

    def search(self, grph, n=1, node_order=None):
        """Search"""
        res = None
        if self.grph is None:
            self.grph = grph
            self.pri = list(self.grph.nodes())
            random.shuffle(self.pri)
            self.cnt = 0
            while "d" in grph.nodes[self.pri[self.cnt]]:
                self.cnt += 1
            res = (-1, -1, -1, -1, self.pri[self.cnt])
            self.cnt += 1
        else:
            while "d" in grph.nodes[self.pri[self.cnt]]:
                self.cnt += 1
            res = (-1, -1, -1, -1, self.pri[self.cnt])
            self.cnt += 1

        return res


class SOFGrph:
    """[NEIGHBOR GRAPH VERSION] submodular function to maximize using coverage"""

    def __init__(self, lmd=1, cdeg=1):
        self.lmd = lmd
        self.cdeg = cdeg

    def evaluate(self, idx, grph):

        cnt_done = 0  # already evaluated
        cnt_cvrd = 0  # already covered

        edges = list(grph.edges(idx))
        cnt_all = len(edges) + 1  # neighbors + self

        nodes = set(itertools.chain.from_iterable(edges)) - {idx}

        for i in nodes:
            if self.cdeg <= 1:
                if "d" in grph.nodes[i]:
                    cnt_done += 1
                if "c" in grph.nodes[i]:
                    cnt_cvrd += 1
            # elif cdeg == 2: """ not implemented"""
            #    if 'd' in grph.nodes[i]:
            #        cnt_done += 1
            #    if 'c2' in grph.nodes[i]:
            #        cnt_cvrd += 1

        return {
            "v": cnt_all - cnt_cvrd - self.lmd * len(list(grph.nodes())) * cnt_done,
            "dup": cnt_done,
            "dup_cvrd": cnt_cvrd,
            "cvrd": cnt_all,
        }


class GreedyAlgoGrph:
    """Greedy Algorithm for Graph"""

    def __init__(self, sof):
        self.sof = sof
        self.start_idx = 0
        self.start_from_previous = False
        if 0.1 < self.sof.lmd:  # speed-up trick for large LMD
            self.start_from_previous = True  # store node_order index as next starting point (work w/ node_order)
            print("[GreedyAlgoGrph] use speed-up trick for LMD({}) is large > 0.1".format(self.sof.lmd))

    def search(self, grph, n=1, node_order=None):

        cnt = 0
        res = None
        _start_idx = self.start_idx
        if node_order is not None:
            for _i, tpl in enumerate(node_order[self.start_idx :]):
                idx = tpl[0]
                # print(tpl, grph.nodes[idx], 'd' in grph.nodes[idx])
                cnt += 1
                if "d" not in grph.nodes[idx]:
                    _res = self.sof.evaluate(idx, grph)
                    if res is None or res[0] < _res["v"]:
                        res = (
                            (_res["v"], _res["dup"], _res["cvrd"], -1, idx)
                            if "dup_cvrd" not in _res.keys()
                            else (_res["v"], _res["dup"], _res["cvrd"], _res["dup_cvrd"], idx)
                        )
                        if self.start_from_previous:  # speed-up trick
                            self.start_idx = _start_idx + _i
                    # print(_res)
                    if ("dup_cvrd" not in _res.keys() and _res["dup"] <= 0) or (
                        "dup_cvrd" in _res.keys() and _res["dup"] <= 0 and _res["dup_cvrd"] <= 0
                    ):
                        # the score is smaller in the later of the list, as the nodes ordered by n edges
                        break

            if res is None:
                for _i, tpl in enumerate(node_order[: self.start_idx]):
                    idx = tpl[0]
                    # print(tpl, grph.nodes[idx], 'd' in grph.nodes[idx])
                    cnt += 1
                    if "d" not in grph.nodes[idx]:
                        _res = self.sof.evaluate(idx, grph)
                        if res is None or res[0] < _res["v"]:
                            res = (
                                (_res["v"], _res["dup"], _res["cvrd"], -1, idx)
                                if "dup_cvrd" not in _res.keys()
                                else (_res["v"], _res["dup"], _res["cvrd"], _res["dup_cvrd"], idx)
                            )
                            if self.start_from_previous:  # speed-up trick
                                self.start_idx = _start_idx + _i
                        # print(_res)
                        if ("dup_cvrd" not in _res.keys() and _res["dup"] <= 0) or (
                            "dup_cvrd" in _res.keys() and _res["dup"] <= 0 and _res["dup_cvrd"] <= 0
                        ):
                            # the score is smaller in the later of the list, as the nodes ordered by n edges
                            break

        return res


def gp_estimate(X, y):
    """Gaussian Process Estimzation"""
    model = GPy.models.GPRegression(
        X=X, Y=y.reshape(len(y), 1), kernel=GPy.kern.RBF(input_dim=len(X[0]), variance=1.0, lengthscale=1.0)
    )
    model.optimize(messages=False, max_iters=1000)
    return model


def gp_predict_scores(gp_model, X, y):
    """Gaussian Process Prediction Scores"""
    _mean, _var = gp_model.predict(X)
    return r2_score(_mean, y), mean_absolute_error(_mean, y)


class BOSample:
    """Bayesian Optimization Sampling"""

    def __init__(self, X_values, y_values, ns, n_first_training, random_seed=0):

        if 0 < n_first_training and len(X_values) == len(y_values) and len(X_values) == len(ns):
            self.all_ns = ns
            self.train_ns = ns[0:n_first_training]
            self.pool_ns = ns[n_first_training:]
            self.train_x = np.array(X_values[0:n_first_training])
            self.train_y = np.array(y_values[0:n_first_training])
            self.pool_x = np.array(X_values[n_first_training:])
            self.pool_y = np.array(y_values[n_first_training:])

            print("ns:", len(self.all_ns), len(self.train_ns), len(self.pool_ns))
            print("ds:", len(self.train_x), len(self.train_y), len(self.pool_x), len(self.pool_y))

            self.model = self.estimate_model(self.train_x, self.train_y)
            self.means, self.vars = self.model.predict(self.pool_x)

        else:
            print("X, y, ns:", len(X_values), len(y_values), len(ns))
            self.all_ns = ns
            self.train_ns = []
            self.pool_ns = ns
            self.train_x = None
            self.train_y = None
            self.pool_x = np.array(X_values)
            self.pool_y = np.array(y_values)

            print("ns:", len(self.all_ns), len(self.train_ns), len(self.pool_ns))
            print("ds:", len(self.train_x), len(self.train_y), len(self.pool_x), len(self.pool_y))

            self.model = None
            self.means, self.vars = None, None

        random.seed(random_seed)

    def estimate_model(self, X, y):
        self.model = GPy.models.GPRegression(
            X=X, Y=y.reshape(len(y), 1), kernel=GPy.kern.RBF(input_dim=len(X[0]), variance=1.0, lengthscale=1.0)
        )
        self.model.optimize(messages=False, max_iters=1000)
        return self.model

    def search(self, grph, n=1, node_order=None):

        if self.vars is None:
            max_i = random.randrange(0, len(list(grph.nodes())))
            max_v = [-1]

        else:
            max_i = np.argmax(self.vars)
            max_v = self.vars[max_i]
            #         print(self.vars)
            #         print(max_i, np.max(self.vars), max_v, self.vars[0:10])

        # print(sorted([_v[0] for _v in self.vars], reverse=True)[0:5])

        max_name = self.pool_ns.pop(max_i)
        self.train_ns.append(max_name)

        self.train_x = np.append(self.train_x, [self.pool_x[max_i]], axis=0)
        self.train_y = np.append(self.train_y, self.pool_y[max_i])
        self.pool_x = np.delete(self.pool_x, max_i, axis=0)
        self.pool_y = np.delete(self.pool_y, max_i)

        #         print("ns:", len(self.all_ns), len(self.train_ns), len(self.pool_ns))
        #         print("ds:", len(self.train_x), len(self.train_y), len(self.pool_x), len(self.pool_y))

        self.model = self.estimate_model(self.train_x, self.train_y)
        self.means, self.vars = self.model.predict(self.pool_x)

        return (max_v[0], len(self.train_x), len(self.pool_x), -1, self.all_ns.index(max_name))

    def get_model(self):
        return self.model

    def predict_scores(self, X, y):
        _mean, _var = self.model.predict(X)
        return r2_score(_mean, y), mean_absolute_error(_mean, y)


class SamplerRunner:
    def __init__(self, algo, nb_grph, ns, level_cvrg=2, first_n=0):
        self.algo = algo
        self.nb_grph = nb_grph
        self.ns = ns
        self.level_cvrg = level_cvrg
        self.first_n = first_n

        NBGraphChecker.set_first_n_nodes_as_done(self.nb_grph, self.first_n, self.ns)
        self.covered_nodes = NBGraphChecker.create_empty_covered_nodes()
        self.nb_sorted_by_degree = NBGraphChecker.nb_sorted_by_degree(self.nb_grph)
        NBGraphChecker.pop_first_n_covered_nodes(
            self.nb_grph, self.nb_sorted_by_degree, self.ns, self.first_n, self.covered_nodes, level_cvrg=self.level_cvrg
        )
        self.res = []

    def run(self, nrun):

        cnt = 0
        while cnt < nrun:
            res = self.algo.search(self.nb_grph, node_order=self.nb_sorted_by_degree)
            NBGraphChecker.store_search(self.nb_grph, 0, res[0], res[1], res[2], res[3], res[4], self.ns)
            (cvrg1, cvrg2) = NBGraphChecker.pop_node_update_covered_nodes(
                self.nb_grph, self.nb_sorted_by_degree, self.ns, res[4], self.covered_nodes, level_cvrg=self.level_cvrg
            )
            self.res.append(
                {
                    "idx": res[4],
                    "v": res[0],
                    "n_done": res[1],
                    "n_nyet": res[2],
                    "n_cvrded": res[3],
                    "cvrg1": cvrg1,
                    "cvrg2": cvrg2,
                }
            )

            cnt += 1

        return self.res
