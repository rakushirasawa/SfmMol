import collections
import itertools
import os
import pickle
import random
import statistics

import networkx as nx
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdCoordGen
from scipy.stats import kurtosis, skew


class MolLoader:
    @classmethod
    def load(cls, mol_paths, slv_path=None, n_sample=-1):
        slv_ms = []
        slv_ns = []
        slv_ss = []

        if slv_path is not None:
            print("Load solvents")
            with open(slv_path) as f:
                ls = f.readlines()
                for i, l in enumerate(ls):
                    print("{:20}".format("  cnt: {} (total: {})".format(i + 1, len(ls))), end="\r")
                    m = Chem.MolFromSmiles(l.split()[0])
                    if m is not None:
                        m.RemoveAllConformers()
                        rdCoordGen.AddCoords(m)
                        slv_ms.append(m)
                        slv_ns.append(l.split()[1])
                        slv_ss.append(Chem.MolToSmiles(m))

            print("")

        ls = []
        dup_ns = []
        for path in mol_paths:
            print("Load generated mols from", path)
            with open(path) as f:
                ls += f.readlines()
                print("total:", len(ls))

        if n_sample > 0:
            random.seed(1)
            ls = random.sample(ls, n_sample)
            print("  sampled:", len(ls))

        ms = []
        ns = []
        ss = []
        for i, l in enumerate(ls):
            print("{:20}".format("  cnt: {} (total: {})".format(i + 1, len(ls))), end="\r")
            m = Chem.MolFromSmiles(l.split()[0])
            if m is not None:
                s = Chem.MolToSmiles(m)
                if s in ss:
                    idx = ss.index(s)
                    ns[idx] = ns[idx] + "_" + l.split()[1]
                    dup_ns.append(ns[idx])
                else:
                    m.RemoveAllConformers()
                    rdCoordGen.AddCoords(m)
                    ms.append(m)
                    ns.append(l.split()[1])
                    ss.append(s)

        print("")
        print("number of mols:", len(ms))
        print("duplicate mols:", dup_ns)

        return (ms, ns, slv_ms, slv_ns)

    @classmethod
    def calc_fp(cls, ms, radius, nbits, slv_ms=None, slv_radius=-1, slv_nbits=-1, slv_fp_n_extend=1, bifs=None, slv_bifs=None):

        print("[Molecul]ECFP{}({})".format(radius * 2, nbits))
        fps = []
        for m in ms:
            _binfo = {}
            fp = AllChem.GetMorganFingerprintAsBitVect(m, radius, nbits, bitInfo=_binfo)
            fps.append(fp)
            if bifs is not None:
                bifs.append(_binfo)

        comb_fps = []

        if slv_ms is not None:

            print("[Solvent]ECFP{}({},x{})".format(slv_radius * 2, slv_nbits, slv_fp_n_extend))
            slv_fps = []
            for m in slv_ms:
                if m is not None:
                    _binfo = {}
                    fp = AllChem.GetMorganFingerprintAsBitVect(m, slv_radius, slv_nbits, bitInfo=_binfo)
                    # fp = AllChem.GetMorganFingerprint(m, SLV_RADIUS)
                    slv_fps.append(fp)
                    if slv_bifs is not None:
                        slv_bifs.append(_binfo)
                else:
                    slv_fps.append(None)
                    slv_bifs.append(None)

            for fp in fps:
                for slv_fp in slv_fps:
                    #
                    # bit extension
                    #
                    if slv_fp is None:
                        comb_fps.append(fp)
                    elif slv_fp_n_extend == 1:
                        comb_fps.append(fp + slv_fp)
                    elif slv_fp_n_extend == 2:
                        comb_fps.append(fp + slv_fp + slv_fp)  # x2
                    elif slv_fp_n_extend == 3:
                        comb_fps.append(fp + slv_fp + slv_fp + slv_fp)  # x3
                    elif slv_fp_n_extend == 4:
                        comb_fps.append(fp + slv_fp + slv_fp + slv_fp + slv_fp)  # x4
                    elif slv_fp_n_extend == 5:
                        comb_fps.append(fp + slv_fp + slv_fp + slv_fp + slv_fp + slv_fp)  # x5
                    elif slv_fp_n_extend == 6:
                        comb_fps.append(fp + slv_fp + slv_fp + slv_fp + slv_fp + slv_fp + slv_fp)  # x6
                    elif slv_fp_n_extend == 8:
                        comb_fps.append(fp + slv_fp + slv_fp + slv_fp + slv_fp + slv_fp + slv_fp + slv_fp + slv_fp)  # x8
                    elif slv_fp_n_extend == 16:
                        comb_fps.append(
                            fp
                            + slv_fp
                            + slv_fp
                            + slv_fp
                            + slv_fp
                            + slv_fp
                            + slv_fp
                            + slv_fp
                            + slv_fp
                            + slv_fp
                            + slv_fp
                            + slv_fp
                            + slv_fp
                            + slv_fp
                            + slv_fp
                            + slv_fp
                            + slv_fp
                        )
                    else:
                        raise RuntimeError("slv_fp_n_extend={} is not supported".format(slv_fp_n_extend))

        else:
            comb_fps = fps

        return comb_fps

    @classmethod
    def calc_joint_fp(
        cls, ms, ns, radius, nbits, slv_ms, slv_ns, slv_radius=-1, slv_nbits=-1, slv_fp_n_extend=1, bifs=None, slv_bifs=None
    ):

        comb_ns = []
        for n in ns:
            for slv_n in slv_ns:
                comb_ns.append(n + "_" + slv_n[0:6])

        comb_fps = MolLoader.calc_fp(
            ms,
            radius,
            nbits,
            slv_ms,
            slv_radius=slv_radius,
            slv_nbits=slv_nbits,
            slv_fp_n_extend=slv_fp_n_extend,
            bifs=bifs,
            slv_bifs=slv_bifs,
        )

        return (comb_ns, comb_fps)

    @classmethod
    def calc_dm(cls, fps):
        _dis_matrix = []
        for idx in range(len(fps)):
            _dis_matrix.append(DataStructs.BulkTanimotoSimilarity(fps[idx], fps[: len(fps)], returnDistance=True))

        return np.array(_dis_matrix)


class NBGraphChecker:
    """NBGraphChecker
    Check Neighbor Graph Statistics
    """

    @classmethod
    def check(cls, nb_grph, r_cvrg=1, n_cvrg=10, calc_avg_path_len=False):
        """calc from population"""
        _dg_nd = sorted([(d, n) for n, d in nb_grph.degree()], reverse=True)
        _nd = [dn[1] for dn in _dg_nd]
        _dg = [dn[0] for dn in _dg_nd]

        # print(len(nb_grph.nodes()))

        _n_edges = len(nb_grph.edges())
        _clustering = nx.average_clustering(nb_grph)
        _n_connected_components = len([0 for _ in nx.connected_components(nb_grph)])
        _avg_path_len = None
        print("  clustering coef. (random=>0.0):", _clustering)
        print("  n connected components:", _n_connected_components)
        # _g_random = nx.connected_watts_strogatz_graph(len(nb_grph.nodes()), 2, 1)
        # print('cluster(random):', nx.average_clustering(_g_random))
        if calc_avg_path_len and nx.is_connected(nb_grph):
            _avg_path_len = nx.average_shortest_path_length(nb_grph)
            print("  avg_path_len(nb_grph)", _avg_path_len)
            # _g_random = nx.connected_watts_strogatz_graph(len(nb_grph.nodes()), 2, 1)
            # print('  avg_path_len(random)', nx.average_shortest_path_length(_g_random))

        _mean = statistics.mean(_dg)
        _median = statistics.median(_dg)
        _std = statistics.stdev(_dg)
        _skew = skew(_dg)
        _kurt = kurtosis(_dg)
        _max = max(_dg)
        _min = min(_dg)
        _deg = collections.Counter(_dg)

        # calculate coverage
        _n_cvrg = n_cvrg
        if len(_nd) < _n_cvrg:
            _n_cvrg = len(_nd)

        _cvrg = None
        if r_cvrg < 0:  # skipp coverage calculation
            _cvrg = None
        elif r_cvrg == 1:
            _cvrg = {
                i: 100 * (len(set(itertools.chain.from_iterable(nb_grph.edges(_nd[0 : i + 1])))) / len(_nd))
                for i in range(_n_cvrg)
            }  # r = 1
        elif r_cvrg == 2:
            _cvrg = {
                i: 100
                * (
                    len(
                        set(
                            itertools.chain.from_iterable(
                                nb_grph.edges(set(itertools.chain.from_iterable(nb_grph.edges(_nd[0 : i + 1]))))
                            )
                        )
                    )
                    / len(_nd)
                )
                for i in range(_n_cvrg)
            }  # r = 2
        elif r_cvrg == 3:
            _cvrg = {
                i: 100
                * (
                    len(
                        set(
                            itertools.chain.from_iterable(
                                nb_grph.edges(
                                    set(
                                        itertools.chain.from_iterable(
                                            nb_grph.edges(set(itertools.chain.from_iterable(nb_grph.edges(_nd[0 : i + 1]))))
                                        )
                                    )
                                )
                            )
                        )
                    )
                    / len(_nd)
                )
                for i in range(_n_cvrg)
            }  # r = 3
        elif r_cvrg == 4:
            _cvrg = {
                i: 100
                * (
                    len(
                        set(
                            itertools.chain.from_iterable(
                                nb_grph.edges(
                                    set(
                                        itertools.chain.from_iterable(
                                            nb_grph.edges(
                                                set(
                                                    itertools.chain.from_iterable(
                                                        nb_grph.edges(
                                                            set(itertools.chain.from_iterable(nb_grph.edges(_nd[0 : i + 1])))
                                                        )
                                                    )
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                    / len(_nd)
                )
                for i in range(_n_cvrg)
            }  # r = 3
        else:
            raise RuntimeError("r_cvrg({}) is not supported. only 1-3".format(r_cvrg))

        return (
            _mean,
            _median,
            _std,
            _max,
            _min,
            _skew,
            _kurt,
            _deg,
            _cvrg,
            _n_edges,
            _clustering,
            _n_connected_components,
            _avg_path_len,
        )

    @classmethod
    def nb_sorted_by_degree(cls, nb_grph):
        """nb_sorted_by_degree"""
        nb_sorted_by_degree = sorted(nb_grph.degree, key=lambda x: x[1], reverse=True)
        return nb_sorted_by_degree

    @classmethod
    def store_search(cls, nb_grph, cnt, score, n_done, n_nyet, n_cvrded, idx, ns):
        """store search result"""
        nb_grph.nodes[idx]["d"] = (cnt, score, n_done, n_nyet, n_cvrded, idx, ns[idx])

    @classmethod
    def create_empty_covered_nodes(cls, level=2):
        return [set(), set()]

    @classmethod
    def pop_first_n_covered_nodes(cls, nb_grph, nb_sorted_by_degree, ns, nnode, covered_nodes, level_cvrg=0):
        """calc coverage"""
        if 0 < nnode:
            cnt = 0
            LIST_LEN = len(nb_sorted_by_degree)
            for i, v in enumerate(reversed(nb_sorted_by_degree)):
                # print(v[0],end=', ')
                if v[0] < nnode:
                    # print(cnt,LIST_LEN-i-1,v[0])
                    # _node = nb_sorted_by_degree.pop(LIST_LEN - i - 1)
                    nb_sorted_by_degree.pop(LIST_LEN - i - 1)
                    cnt += 1
                    if 1 <= level_cvrg:
                        _newly_covered = {v[0]} | set(itertools.chain.from_iterable(list(nb_grph.edges(v[0]))))
                        covered_nodes[0] |= _newly_covered
                        if 2 <= level_cvrg:
                            covered_nodes[1] |= _newly_covered
                            #                             for l in itertools.chain.from_iterable(list(_newly_covered)):
                            for _l in list(_newly_covered):
                                covered_nodes[1] |= set(itertools.chain.from_iterable(list(nb_grph.edges(_l))))

                    if nnode <= cnt:
                        # print(cnt,LIST_LEN-i,v[0])
                        if 1 <= level_cvrg:
                            for j in range(nnode):
                                nb_grph.nodes[j]["d"] += (len(covered_nodes[0]) / len(ns),)
                                if 2 <= level_cvrg:
                                    nb_grph.nodes[j]["d"] += (len(covered_nodes[1]) / len(ns),)
                                    # print(nb_grph.nodes[j]['d'])
                            for j in covered_nodes[0]:
                                nb_grph.nodes[j]["c"] = 1
                            for j in covered_nodes[1]:
                                nb_grph.nodes[j]["c2"] = 1
                        break

        # dummy values for regression performance (R2, MAE) = (nan, nan)
        for i in range(nnode):
            nb_grph.nodes[i]["d"] += ("nan",)
            nb_grph.nodes[i]["d"] += ("nan",)

        return covered_nodes

    @classmethod
    def pop_node_update_covered_nodes(cls, nb_grph, nb_sorted_by_degree, ns, idx, covered_nodes, level_cvrg=0):
        """calc coverage"""
        cvrg1 = -1
        cvrg2 = -1
        for _i, _v in enumerate(nb_sorted_by_degree):
            if _v[0] == idx:
                nb_sorted_by_degree.pop(_i)
                if 1 <= level_cvrg:
                    _newly_covered_by_idx = {idx} | set(itertools.chain.from_iterable(list(nb_grph.edges(idx))))
                    covered_nodes[0] |= _newly_covered_by_idx
                    cvrg1 = len(covered_nodes[0]) / len(ns)
                    nb_grph.nodes[idx]["d"] += (cvrg1,)
                    # set covered as 'c' = 1
                    for _j in _newly_covered_by_idx:
                        nb_grph.nodes[_j]["c"] = 1

                    if 2 <= level_cvrg:
                        covered_nodes[1] |= _newly_covered_by_idx
                        #                             for l in itertools.chain.from_iterable(list(_newly_covered_by_idx)):
                        for _l in list(_newly_covered_by_idx):
                            _newly_covered_2nd_by_idx = {_l} | set(itertools.chain.from_iterable(list(nb_grph.edges(_l))))
                            covered_nodes[1] |= _newly_covered_2nd_by_idx
                            # set covered(2nd) as 'c2' = 1
                            for _j in _newly_covered_2nd_by_idx:
                                nb_grph.nodes[_j]["c2"] = 1
                        cvrg2 = len(covered_nodes[1]) / len(ns)
                        nb_grph.nodes[idx]["d"] += (cvrg2,)
                break

        return (cvrg1, cvrg2)

    @classmethod
    def set_first_n_nodes_as_done(cls, nb_grph, nnode, ns):
        for idx in range(nnode):
            nb_grph.nodes[idx]["d"] = (-1, -1, -1, -1, -1, idx, ns[idx])


class NBGraphMaker:
    """NBGraphMaker
    Make Neighbor Graph from Fingerprint
    """

    @classmethod
    def create(cls, fps, thresh, rm_isolates=False, file_name=None):

        nb_grph = nx.Graph()
        nb_grph.add_nodes_from([i for i in range(len(fps))])
        for i, fp in enumerate(fps):
            print("  gen edges for nodes (th={}): {} (total:{}){:20}".format(thresh, i + 1, len(fps), ""), end="\r")
            _dis_mtx = DataStructs.BulkTanimotoSimilarity(fp, fps, returnDistance=True)
            for j, d in enumerate(_dis_mtx):
                if d < thresh and (i != j):
                    nb_grph.add_edge(i, j)
        print("")
        print("  egdes:", len(nb_grph.edges()))

        if rm_isolates:
            print("  isolates: {} removed".format(len(list(nx.isolates(nb_grph)))))
            print("     nodes: {}".format(len(nb_grph.nodes())))
            nb_grph.remove_nodes_from(list(nx.isolates(nb_grph)))
            print("     nodes: {}".format(len(nb_grph.nodes())))

        if file_name is not None:
            file_path = os.path.dirname(file_name)
            if not os.path.exists(file_path):
                os.makedirs(file_path)
            with open(file_name, "wb") as f:
                pickle.dump((thresh, nb_grph), f)
            print("  nb_grph saved in", file_name)

        return nb_grph

    @classmethod
    def load(cls, file_name):
        with open(file_name, "rb") as f:
            print("nb_file:", file_name)
            return pickle.load(f)
