# from alotustools.aggutils import rf_estimate, rf_predict_scores
import collections
import pickle

import pytest
from pytest import approx

from sfmmol import (
    BOSample,
    GreedyAlgoGrph,
    MolLoader,
    NBGraphChecker,
    NBGraphMaker,
    RandomSampleGrph,
    SamplerRunner,
    SOFGrph,
    StratifiedRandomSampleGrph,
)

delta = 1e-2


@pytest.mark.parametrize(
    "mpaths, spath, prefix, expects",
    [
        (
            ["data/processed/ACCESSIBLE.smi"],
            "data/processed/SOLVENTS.smi",
            None,
            (
                10,
                16,
                '"D2860_87558538_5,15-Diphenylporphyrin"',
                "01-Water",
                0.4,
                0.0,
                0.699206,
                1.39792,
                0.6115702,
                2,
                0,
                (3, 2),
            ),
        ),
        (
            ["data/processed/ACCESSIBLE.smi", "data/processed/GENERATED.smi"],
            "data/processed/SOLVENTS.smi",
            "data/processed/d_",
            (
                5920,
                16,
                '"D2860_87558538_5,15-Diphenylporphyrin"',
                "01-Water",
                9.396959,
                5.0,
                13.1488621,
                3.2216427,
                15.08408638,
                141,
                0,
                (28, 141),
            ),
        ),
    ],
)
def test_molloader(mpaths, spath, prefix, expects):

    mols_file_name = None if prefix is None else f"{prefix}mols.pkl"
    nb_grph_file_name = None if prefix is None else f"{prefix}nb_grph.pkl"

    if mpaths != ["data/processed/ACCESSIBLE.smi", "data/processed/GENERATED.smi"]:
        # if True:
        """Load Molecule"""
        (ms, ns, slv_ms, slv_ns) = MolLoader.load(mol_paths=mpaths, slv_path=spath, n_sample=-1)
        assert len(ms) == expects[0]
        assert len(ns) == expects[0]
        assert len(slv_ms) == expects[1]
        assert len(slv_ns) == expects[1]
        assert ns[0] == expects[2]
        assert slv_ns[0] == expects[3]

        """Calc Fingerprint"""
        RADIUS = 3
        NBITS = 2048
        fps = MolLoader.calc_fp(ms, RADIUS, NBITS)
        assert len(fps) == expects[0]

        """Calc Distance Matrix"""
        dm = MolLoader.calc_dm(fps)
        assert dm.shape == (expects[0], expects[0])

        """Calc Fingerprint for MOL & SLV"""
        SLV_RADIUS = 2
        SLV_NBITS = 128
        BIT_EXTEND = 1
        (comb_ns, comb_fps) = MolLoader.calc_joint_fp(
            ms, ns, RADIUS, NBITS, slv_ms, slv_ns, SLV_RADIUS, SLV_NBITS, slv_fp_n_extend=BIT_EXTEND
        )
        assert len(comb_fps) == expects[0] * expects[1]
        assert len(comb_ns) == expects[0] * expects[1]

        """Save/Load objects"""
        if mols_file_name is not None:
            with open(mols_file_name, "wb") as f:
                pickle.dump((ms, ns, fps, dm, slv_ms, slv_ns, comb_ns, comb_fps), f)

        """NB Graph"""
        thresh = 0.3
        nb_grph = NBGraphMaker.create(fps, thresh, file_name=nb_grph_file_name)

    else:
        with open(mols_file_name, "rb") as f:
            (ms, ns, fps, dm, slv_ms, slv_ns, comb_ns, comb_fps) = pickle.load(f)

        (thresh, nb_grph) = NBGraphMaker.load(nb_grph_file_name)

    (
        _g_mean,
        _g_median,
        _g_std,
        _g_max,
        _g_min,
        _g_skew,
        _g_kurt,
        _deg,
        _cvrg,
        _n_edges,
        _clustering,
        _n_connected_components,
        _avg_path_len,
    ) = NBGraphChecker.check(nb_grph, r_cvrg=-1, n_cvrg=-1)
    print(
        f"NBGrph: mean({_g_mean}), median({_g_median}), stdev({_g_std}),",
        f"skew({_g_skew}), kurt({_g_kurt}), max({_g_max}), min({_g_min})",
    )
    assert _g_mean == approx(expects[4], abs=delta)
    assert _g_median == approx(expects[5], abs=delta)
    assert _g_std == approx(expects[6], abs=delta)
    assert _g_skew == approx(expects[7], abs=delta)
    assert _g_kurt == approx(expects[8], abs=delta)
    assert _g_max == expects[9]
    assert _g_min == expects[10]

    nb_sorted_by_degree = NBGraphChecker.nb_sorted_by_degree(nb_grph)
    assert len(nb_sorted_by_degree) == expects[0]
    assert nb_sorted_by_degree[0] == expects[11]


@pytest.mark.parametrize(
    "rnd_seed, mols_pkl, nb_grph_pkl, expects",
    [
        (0, "data/processed/d_mols.pkl", "data/processed/d_nb_grph.pkl", (147, 5573)),
        (1, "data/processed/d_mols.pkl", "data/processed/d_nb_grph.pkl", (1695, 4982)),
    ],
)
def test_random(rnd_seed, mols_pkl, nb_grph_pkl, expects):

    with open(mols_pkl, "rb") as f:
        (_dmy, ns, _dmy, _dmy, _dmy, _dmy, _dmy, _dmy) = pickle.load(f)

    algo = RandomSampleGrph(random_seed=rnd_seed)
    (thresh, nb_grph) = NBGraphMaker.load(nb_grph_pkl)
    nb_sorted_by_degree = NBGraphChecker.nb_sorted_by_degree(nb_grph)

    res = algo.search(nb_grph, node_order=nb_sorted_by_degree)
    idx = res[4]
    # print(idx, ns[idx])
    assert res[4] == expects[0]

    NBGraphChecker.store_search(nb_grph, 0, res[0], res[1], res[2], res[3], res[4], ns)
    covered_nodes = NBGraphChecker.create_empty_covered_nodes()
    NBGraphChecker.pop_node_update_covered_nodes(nb_grph, nb_sorted_by_degree, ns, idx, covered_nodes, level_cvrg=2)

    res = algo.search(nb_grph, node_order=nb_sorted_by_degree)
    idx = res[4]
    # print(idx, ns[idx])
    assert res[4] == expects[1]

    NBGraphChecker.store_search(nb_grph, 0, res[0], res[1], res[2], res[3], res[4], ns)
    NBGraphChecker.pop_node_update_covered_nodes(nb_grph, nb_sorted_by_degree, ns, idx, covered_nodes, level_cvrg=2)

    """
    cnt = 0
    while cnt < 10:
        res = algo.search(nb_grph, node_order=nb_sorted_by_degree)
        idx = res[4]
        # print(idx, ns[idx])
        nbchkr.store_search(nb_grph, 0, res[0], res[1], res[2], res[3], res[4], ns)
        covered_nodes = nbchkr.create_empty_covered_nodes()
        nbchkr.pop_node_update_covered_nodes(nb_grph, nb_sorted_by_degree, ns, idx, covered_nodes, level_cvrg=2)
        cnt += 1
    """


@pytest.mark.parametrize(
    "rnd_seed, mols_pkl, strada, nb_grph_pkl, expects",
    [
        (
            0,
            "data/processed/d_mols.pkl",
            {"^01_": 1699, "^02_": 1700, "^03_": 717, "^04_": 1699, "^05_": 103},
            "data/processed/d_nb_grph.pkl",
            (1765, 2702),
        ),
        (
            1,
            "data/processed/d_mols.pkl",
            {"^01_": 1699, "^02_": 1700, "^03_": 717, "^04_": 1699, "^05_": 103},
            "data/processed/d_nb_grph.pkl",
            (2482, 1695),
        ),
    ],
)
def test_stratified_random(rnd_seed, mols_pkl, strada, nb_grph_pkl, expects):

    with open(mols_pkl, "rb") as f:
        (_dmy, ns, _dmy, _dmy, _dmy, _dmy, _dmy, _dmy) = pickle.load(f)

    algo = StratifiedRandomSampleGrph(strada=strada, ns=ns, random_seed=rnd_seed)
    (thresh, nb_grph) = NBGraphMaker.load(nb_grph_pkl)
    nb_sorted_by_degree = NBGraphChecker.nb_sorted_by_degree(nb_grph)

    res = algo.search(nb_grph, node_order=nb_sorted_by_degree)
    idx = res[4]
    # print(idx, ns[idx])
    assert res[4] == expects[0]

    NBGraphChecker.store_search(nb_grph, 0, res[0], res[1], res[2], res[3], res[4], ns)
    covered_nodes = NBGraphChecker.create_empty_covered_nodes()
    NBGraphChecker.pop_node_update_covered_nodes(nb_grph, nb_sorted_by_degree, ns, idx, covered_nodes, level_cvrg=2)

    res = algo.search(nb_grph, node_order=nb_sorted_by_degree)
    idx = res[4]
    # print(idx, ns[idx])
    assert res[4] == expects[1]

    NBGraphChecker.store_search(nb_grph, 0, res[0], res[1], res[2], res[3], res[4], ns)
    NBGraphChecker.pop_node_update_covered_nodes(nb_grph, nb_sorted_by_degree, ns, idx, covered_nodes, level_cvrg=2)

    """
    cnt = 0
    while cnt < 10:
        res = algo.search(nb_grph, node_order=nb_sorted_by_degree)
        idx = res[4]
        # print(idx, ns[idx])
        nbchkr.store_search(nb_grph, 0, res[0], res[1], res[2], res[3], res[4], ns)
        nbchkr.pop_node_update_covered_nodes(nb_grph, nb_sorted_by_degree, ns, idx, covered_nodes, level_cvrg=2)
        cnt += 1
    """


@pytest.mark.parametrize(
    "rnd_seed, mols_pkl, label_pkl, nb_grph_pkl, expects",
    [
        (
            0,
            "data/processed/d_mols.pkl",
            "data/processed/d_labels.pkl",
            "data/processed/d_nb_grph.pkl",
            (147, 796),
        ),
        (
            1,
            "data/processed/d_mols.pkl",
            "data/processed/d_labels.pkl",
            "data/processed/d_nb_grph.pkl",
            (2482, 4982),
        ),
    ],
)
def test_knn_stratified_random(rnd_seed, mols_pkl, label_pkl, nb_grph_pkl, expects):

    with open(mols_pkl, "rb") as f:
        (_dmy, ns, fps, _dmy, _dmy, _dmy, _dmy, _dmy) = pickle.load(f)

    with open(label_pkl, "rb") as f:
        mol_strada = pickle.load(f)
        ns_strada = [f"{_stratum}_{_name}" for _stratum, _name in zip(mol_strada.labels_, ns)]
        strada = dict(
            [(f"^{key}_", val) for key, val in dict(sorted(dict(collections.Counter(mol_strada.labels_)).items())).items()]
        )

    algo = StratifiedRandomSampleGrph(strada=strada, ns=ns_strada, random_seed=rnd_seed, printlevel=1)
    (thresh, nb_grph) = NBGraphMaker.load(nb_grph_pkl)
    nb_sorted_by_degree = NBGraphChecker.nb_sorted_by_degree(nb_grph)

    res = algo.search(nb_grph, node_order=nb_sorted_by_degree)
    idx = res[4]
    print(idx, ns[idx])
    assert res[4] == expects[0]

    NBGraphChecker.store_search(nb_grph, 0, res[0], res[1], res[2], res[3], res[4], ns)
    covered_nodes = NBGraphChecker.create_empty_covered_nodes()
    NBGraphChecker.pop_node_update_covered_nodes(nb_grph, nb_sorted_by_degree, ns, idx, covered_nodes, level_cvrg=2)

    res = algo.search(nb_grph, node_order=nb_sorted_by_degree)
    idx = res[4]
    print(idx, ns[idx])
    assert res[4] == expects[1]

    NBGraphChecker.store_search(nb_grph, 0, res[0], res[1], res[2], res[3], res[4], ns)
    NBGraphChecker.pop_node_update_covered_nodes(nb_grph, nb_sorted_by_degree, ns, idx, covered_nodes, level_cvrg=2)

    """
    cnt = 0
    while cnt < 10:
        res = algo.search(nb_grph, node_order=nb_sorted_by_degree)
        idx = res[4]
        # print(idx, ns[idx])
        nbchkr.store_search(nb_grph, 0, res[0], res[1], res[2], res[3], res[4], ns)
        nbchkr.pop_node_update_covered_nodes(nb_grph, nb_sorted_by_degree, ns, idx, covered_nodes, level_cvrg=2)
        cnt += 1
    """


@pytest.mark.parametrize(
    "lmd, mols_pkl, nb_grph_pkl, expects",
    [
        (
            0,
            "data/processed/d_mols.pkl",
            "data/processed/d_nb_grph.pkl",
            (49, 2125, 395),
        ),
        (
            0.00001,
            "data/processed/d_mols.pkl",
            "data/processed/d_nb_grph.pkl",
            (49, 2125, 1825),
        ),
        (
            1,
            "data/processed/d_mols.pkl",
            "data/processed/d_nb_grph.pkl",
            (49, 2125, 2417),
        ),
    ],
)
def test_sfm(lmd, mols_pkl, nb_grph_pkl, expects):

    with open(mols_pkl, "rb") as f:
        (_dmy, ns, _dmy, _dmy, _dmy, _dmy, _dmy, _dmy) = pickle.load(f)

    (thresh, nb_grph) = NBGraphMaker.load(nb_grph_pkl)
    NBGraphChecker.set_first_n_nodes_as_done(nb_grph, 10, ns)
    covered_nodes = NBGraphChecker.create_empty_covered_nodes()
    nb_sorted_by_degree = NBGraphChecker.nb_sorted_by_degree(nb_grph)
    NBGraphChecker.pop_first_n_covered_nodes(nb_grph, nb_sorted_by_degree, ns, 10, covered_nodes, level_cvrg=2)

    algo = GreedyAlgoGrph(SOFGrph(lmd=lmd))
    res = algo.search(nb_grph, node_order=nb_sorted_by_degree)
    idx = res[4]
    print(idx, ns[idx], nb_sorted_by_degree[0:10])
    assert res[4] == expects[0]
    NBGraphChecker.store_search(nb_grph, 0, res[0], res[1], res[2], res[3], res[4], ns)
    NBGraphChecker.pop_node_update_covered_nodes(nb_grph, nb_sorted_by_degree, ns, idx, covered_nodes, level_cvrg=2)

    res = algo.search(nb_grph, node_order=nb_sorted_by_degree)
    idx = res[4]
    print(idx, ns[idx], nb_sorted_by_degree[0:10])
    assert res[4] == expects[1]
    NBGraphChecker.store_search(nb_grph, 0, res[0], res[1], res[2], res[3], res[4], ns)
    NBGraphChecker.pop_node_update_covered_nodes(nb_grph, nb_sorted_by_degree, ns, idx, covered_nodes, level_cvrg=2)

    cnt = 0
    while cnt < 30:
        res = algo.search(nb_grph, node_order=nb_sorted_by_degree)
        idx = res[4]
        # print(idx, ns[idx])
        NBGraphChecker.store_search(nb_grph, 0, res[0], res[1], res[2], res[3], res[4], ns)
        NBGraphChecker.pop_node_update_covered_nodes(nb_grph, nb_sorted_by_degree, ns, idx, covered_nodes, level_cvrg=2)
        cnt += 1

    assert res[4] == expects[2]


@pytest.mark.parametrize(
    "xy_pkl, mols_pkl, nb_grph_pkl, expects",
    [
        (
            "data/processed/d_xy.pkl",
            "data/processed/d_mols.pkl",
            "data/processed/d_nb_grph.pkl",
            (4133, 5142, 1604, 0.00878438, 0.0420608),
        ),
    ],
)
def test_bo(xy_pkl, mols_pkl, nb_grph_pkl, expects):

    with open(xy_pkl, "rb") as f:
        (lps, fparr) = pickle.load(f)

    with open(mols_pkl, "rb") as f:
        (_dmy, ns, _dmy, _dmy, _dmy, _dmy, _dmy, _dmy) = pickle.load(f)

    FIRST_N = 10
    (thresh, nb_grph) = NBGraphMaker.load(nb_grph_pkl)
    NBGraphChecker.set_first_n_nodes_as_done(nb_grph, FIRST_N, ns)
    covered_nodes = NBGraphChecker.create_empty_covered_nodes()
    nb_sorted_by_degree = NBGraphChecker.nb_sorted_by_degree(nb_grph)
    NBGraphChecker.pop_first_n_covered_nodes(nb_grph, nb_sorted_by_degree, ns, FIRST_N, covered_nodes, level_cvrg=2)

    algo = BOSample(X_values=fparr, y_values=lps, ns=ns, n_first_training=FIRST_N)
    res = algo.search(nb_grph, node_order=nb_sorted_by_degree)
    idx = res[4]
    print(idx, ns[idx], res)
    assert res[4] == expects[0]
    NBGraphChecker.store_search(nb_grph, 0, res[0], res[1], res[2], res[3], res[4], ns)
    NBGraphChecker.pop_node_update_covered_nodes(nb_grph, nb_sorted_by_degree, ns, idx, covered_nodes, level_cvrg=2)

    res = algo.search(nb_grph, node_order=nb_sorted_by_degree)
    idx = res[4]
    print(idx, ns[idx], res)
    assert res[4] == expects[1]
    NBGraphChecker.store_search(nb_grph, 0, res[0], res[1], res[2], res[3], res[4], ns)
    NBGraphChecker.pop_node_update_covered_nodes(nb_grph, nb_sorted_by_degree, ns, idx, covered_nodes, level_cvrg=2)

    cnt = 0
    while cnt < 10:
        res = algo.search(nb_grph, node_order=nb_sorted_by_degree)
        idx = res[4]
        # print(idx, ns[idx])
        NBGraphChecker.store_search(nb_grph, 0, res[0], res[1], res[2], res[3], res[4], ns)
        (cvrg1, cvrg2) = NBGraphChecker.pop_node_update_covered_nodes(
            nb_grph, nb_sorted_by_degree, ns, idx, covered_nodes, level_cvrg=2
        )
        cnt += 1

    assert res[4] == expects[2]
    assert cvrg1 == approx(expects[3], abs=delta)
    assert cvrg2 == approx(expects[4], abs=delta)


@pytest.mark.parametrize(
    "algorithm, mols_pkl, nb_grph_pkl, expects",
    [
        (
            "random",
            "data/processed/d_mols.pkl",
            "data/processed/d_nb_grph.pkl",
            (147, 5573, 2809, 0.01570946, 0.09222973),
        ),
        (
            "srandom",
            "data/processed/d_mols.pkl",
            "data/processed/d_nb_grph.pkl",
            (1765, 2702, 4876, 0.01520270, 0.08682432),
        ),
    ],
)
def test_runner(algorithm, mols_pkl, nb_grph_pkl, expects):

    with open(mols_pkl, "rb") as f:
        (_dmy, ns, _dmy, _dmy, _dmy, _dmy, _dmy, _dmy) = pickle.load(f)

    (thresh, nb_grph) = NBGraphMaker.load(nb_grph_pkl)

    algo = None
    if algorithm == "random":
        algo = RandomSampleGrph(random_seed=0)

    elif algorithm == "srandom":
        strada = {"^01_": 1699, "^02_": 1700, "^03_": 717, "^04_": 1699, "^05_": 103}
        algo = StratifiedRandomSampleGrph(strada=strada, ns=ns, random_seed=0)

    runner = SamplerRunner(algo, nb_grph, ns, level_cvrg=2, first_n=10)
    ret = runner.run(nrun=10)

    assert len(ret) == 10
    assert ret[0]["idx"] == expects[0]
    assert ret[1]["idx"] == expects[1]
    assert ret[9]["idx"] == expects[2]
    assert ret[9]["cvrg1"] == approx(expects[3], abs=delta)
    assert ret[9]["cvrg2"] == approx(expects[4], abs=delta)
