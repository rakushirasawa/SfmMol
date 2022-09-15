import math

import numpy as np
import pandas as pd
from scipy.optimize import leastsq, minimize


def detect_peaks(
    # se, _range, _width=20.0, _ignore_if_disconnected_at=[400.0, 400.5], peak_thres=0.0001, max_only=True, verbose=0
    se,
    _range,
    _ignore_if_disconnected_at,
    _width=20.0,
    peak_thres=0.0001,
    max_only=True,
    verbose=0,
):
    """uvdata peak detector (no fitting, simple version)"""
    peaks = []
    peak_hs = []
    _connected_peak = -1
    for idx in se.loc[_range[0] : _range[1]].index:
        if idx < _range[0] + _width:
            continue
        elif _range[1] <= idx + _width:
            if 0 < _connected_peak:
                # set the largest pos as representative peak
                peaks[-1] = se.loc[peaks[-1] : _connected_peak + 1].idxmax()
                peak_hs.append(min(se[peaks[-1]] - se[peaks[-1] - _width], se[peaks[-1]] - se[peaks[-1] + _width]))
            break
        else:
            if se[idx - _width] < se[idx] and se[idx] > se[idx + _width]:
                if idx - 0.5 == _connected_peak:
                    _connected_peak = idx
                else:
                    peaks.append(idx)
                    _connected_peak = idx
            elif 0 < _connected_peak:
                # set the largest pos as representative peak
                peaks[-1] = se.loc[peaks[-1] : _connected_peak + 1].idxmax()
                peak_hs.append(min(se[peaks[-1]] - se[peaks[-1] - _width], se[peaks[-1]] - se[peaks[-1] + _width]))
                _connected_peak = -1

    # filter small peaks
    _to_rm = []
    for i, peak_h in enumerate(peak_hs):
        if 0 < verbose:
            print(peaks[i], peak_h, end=",")
        if peak_h < peak_thres:
            _to_rm.append(i)

    if 0 < len(peaks):
        if 0 < verbose:
            print("")

    for i in sorted(_to_rm, reverse=True):
        peaks.pop(i)
        peak_hs.pop(i)

    # remove mis-detected peaks at 400.0 or 400.5 discontinuous point
    if _range[0] <= _ignore_if_disconnected_at[0] and _ignore_if_disconnected_at[1] <= _range[1]:
        _to_rm = []
        for i, p in enumerate(peaks):
            if (
                _ignore_if_disconnected_at[0] <= p
                and p <= _ignore_if_disconnected_at[1]
                and 10 * abs(se.loc[_ignore_if_disconnected_at[0]] - se.loc[_ignore_if_disconnected_at[0] - 0.5])
                < abs(se.loc[_ignore_if_disconnected_at[1]] - se.loc[_ignore_if_disconnected_at[0]])
            ):
                _to_rm.append(i)

        for i in sorted(_to_rm, reverse=True):
            peaks.pop(i)
            peak_hs.pop(i)

    if max_only and 1 < len(peaks):
        # print(_range, peaks, end=',')
        max_value = max(peak_hs)
        max_index = peak_hs.index(max_value)

        peaks = [peaks[max_index]]
        peak_hs = [max_value]
        # print(peaks)

    return [(_peak, _height) for _peak, _height in zip(peaks, peak_hs)]


def nm2ev(x):
    return 1240 / x


def ev2nm(x):
    return nm2ev(x)


def sigma2fwhm(sig):
    return sig / (2 * math.sqrt(2 * math.log(2)))


def gauss_func1(parameter, x, y):
    a = parameter[0]
    mew = parameter[1]
    sigma = parameter[2]
    residual = y - (a / (np.sqrt(2 * math.pi) * sigma)) * np.exp(-((x - mew) ** 2) / (2 * sigma**2))
    #     return residual
    return np.linalg.norm(residual, ord=2)


def calc_rss(se1, se2):
    """calculate residual sum of squares"""
    residual = se1 - se2
    ret = 0
    for r in residual:
        ret += r**2
    return ret


# def uvdata_level_min0(se, win=[350, 800]):
def uvdata_level_min0(se, win):
    """uvdata level min set to zero"""
    vec = se.sort_index().copy()
    _min = min(vec[win[0] : win[1]])
    for i in vec.index:
        vec[i] -= _min

    return vec


# def uvdata_level(se, pos=[500.5000]):
def uvdata_level(se, pos):
    """uvdata level gap correction at pos-0.5000 nm and pos nm"""
    vec = se.sort_index(ascending=False).copy()
    # print(vec.index)
    pos_s = sorted(pos, reverse=True)
    # print(pos_s.index)
    for p in pos_s:
        idx = list(vec.index).index(p)
        if 0 < idx and idx < len(list(vec.index)) - 1:
            diff = vec.iloc[idx + 1] - vec.iloc[idx] - (vec.iloc[idx] - vec.iloc[idx - 1])
            for i in vec.iloc[idx + 1 :].index:
                vec[i] -= diff
    vec = vec.sort_index()
    return vec


def uvdata_smooth(se, _range, n=5):
    # def uvdata_smooth(se, _range=[340.5000, 600.0000], n=5):
    """uvdata smoothing by moving average"""
    vec = se.sort_index().copy()
    p_s, p_e = _range[0], _range[1]
    idx_s, idx_e = list(vec.index).index(p_s), list(vec.index).index(p_e)

    # print(idx_s,idx_e)

    if idx_s + n <= idx_e:
        # print(len(list(vec.index)[idx_s:idx_e]))
        # print(len(list(vec.iloc[idx_s:idx_e])))
        b = np.ones(n) / n
        new_y = np.convolve(np.array(list(vec.iloc[idx_s:idx_e])), b, mode="same")
        # print(new_y[0:5])
        n_edge = int(n / 2)
        vec_s = pd.Series(
            new_y[n_edge : len(new_y) - n_edge], name=vec.name, index=list(vec.index)[idx_s + n_edge : idx_e - n_edge]
        )
        # print(vec_s)
        # print(vec_s.loc[idx_s:idx_e])
        vec.update(vec_s)
        # print(vec.loc[idx_s:idx_e])
    return vec


# def uvdata_gauss_fit_nm(se, _range=[400.5000, 450.0000], init_params=[1, 0, 1], info=None):
def uvdata_gauss_fit_nm(se, _range, init_params, info=None):
    """uvdata gauss fit"""
    xdata = np.array(se.loc[_range[0] : _range[1]].index)
    ydata = np.array(se.loc[_range[0] : _range[1]].values)

    #     print(xdata.shape,ydata.shape)
    result = leastsq(gauss_func1, init_params, args=(xdata, ydata))
    a = result[0][0]
    mew = result[0][1]
    sigma = result[0][2]
    #     print(result)

    if info is not None:
        info["a"] = result[0][0]
        info["mew"] = result[0][1]
        info["sigma"] = result[0][2]
        info["intensity"] = info["a"] / (np.sqrt(2 * math.pi) * info["sigma"])
        info["fwhm"] = 2 * math.sqrt(2 * math.log(2)) * info["sigma"]

    xdata = np.array(se.index)
    ypred = (a / (np.sqrt(2 * math.pi) * sigma)) * np.exp(-((xdata - mew) ** 2) / (2 * sigma**2))

    return pd.Series(ypred, index=se.index)


# def uvdata_gauss_fit(se, _range=[400.5000, 450.0000], init_params=[1, 1, 5], info=None):
def uvdata_gauss_fit(se, _range, init_params, info=None):
    """uvdata gauss fit (in eV)
    init_params: [a1,mew1,sigma1,...]
    """
    xdata_ = nm2ev(np.array(se.loc[_range[0] : _range[1]].index))[::-1]
    ydata_ = np.array(se.loc[_range[0] : _range[1]].values)[::-1]

    xdata = xdata_
    ydata = ydata_

    bounds = []
    for i in range(len(init_params)):
        if i == 1:
            bounds.append((nm2ev(_range[1]), nm2ev(_range[0] + 10)))
        #             bounds.append((nm2ev(_range[1]),nm2ev(_range[0])))
        elif i == 2:
            # bounds.append((sigma2fwhm(0.07), sigma2fwhm(0.35)))
            bounds.append((sigma2fwhm(0.07), sigma2fwhm(0.5)))
        else:
            bounds.append((0, None))

    init_params_ = [0 for i in range(len(init_params))]
    init_params_[0] = init_params[0]
    init_params_[1] = nm2ev(init_params[1])
    init_params_[2] = nm2ev(init_params[1]) - nm2ev(init_params[1] + init_params[2])

    #     print(xdata.shape,ydata.shape)
    result = minimize(gauss_func1, init_params_, args=(xdata, ydata), method="SLSQP", bounds=bounds)
    """
    bounds: L-BFGS-B, TNC, SLSQP, Powell, and trust-constr
    constraints: COBYLA, SLSQP and trust-constr
    """
    a, mew, sigma = result.x[0], result.x[1], result.x[2]
    if info is not None:
        info["a"], info["mew"], info["sigma"] = result.x[0], result.x[1], result.x[2]
        info["intensity"], info["fwhm"] = (
            info["a"] / (np.sqrt(2 * math.pi) * info["sigma"]),
            2 * math.sqrt(2 * math.log(2)) * info["sigma"],
        )
        info["result"] = result

    xdata_nm = np.array(se.index)
    ypred = (a / (np.sqrt(2 * math.pi) * sigma)) * np.exp(-((nm2ev(xdata_nm) - mew) ** 2) / (2 * sigma**2))

    yred = np.array([ypred]).transpose()
    # print(yred.shape)

    return pd.DataFrame(yred, index=se.index, columns=["gauss"])


def gauss_func2(parameter, x, y):
    a1, mew1, sigma1 = parameter[0], parameter[1], parameter[2]
    a2, mew2, sigma2 = parameter[3], parameter[4], parameter[5]

    residual = (
        y
        - (a1 / (np.sqrt(2 * math.pi) * sigma1)) * np.exp(-((x - mew1) ** 2) / (2 * sigma1**2))
        - (a2 / (np.sqrt(2 * math.pi) * sigma2)) * np.exp(-((x - mew2) ** 2) / (2 * sigma2**2))
    )

    return np.linalg.norm(residual, ord=2)


# def uvdata_gauss2_fit(se, _range=[400.5000, 450.0000], init_params=[1, 1, 5, 0.1, 2, 5], _cons=None, info=None):
def uvdata_gauss2_fit(se, _range, init_params, _cons=None, info=None):
    """uvdata gauss fit (in eV)
    init_params: [a1,mew1,sigma1,...]
    """
    xdata_ = nm2ev(np.array(se.loc[_range[0] : _range[1]].index))[::-1]
    ydata_ = np.array(se.loc[_range[0] : _range[1]].values)[::-1]

    xdata = xdata_
    ydata = ydata_

    bounds = []
    for i in range(len(init_params)):
        if i == 1:
            bounds.append((nm2ev(_range[1]), nm2ev(_range[0] + 20)))
        elif i % 3 == 1:
            bounds.append((nm2ev(_range[1]), nm2ev(_range[0])))
        elif i == 2:
            #             bounds.append((sigma2fwhm(0.07), sigma2fwhm(0.075)))
            #             bounds.append((sigma2fwhm(0.07), sigma2fwhm(0.09)))
            bounds.append((sigma2fwhm(0.07), sigma2fwhm(0.5)))
        #             bounds.append((sigma2fwhm(0.07), sigma2fwhm(0.09)))
        #             bounds.append((0.01,0.09/(2*math.sqrt(2*math.log(2)))))
        elif i % 3 == 2:
            bounds.append((sigma2fwhm(0.07), sigma2fwhm(0.2)))
        else:
            bounds.append((0, None))

    cons = (
        {"type": "ineq", "fun": lambda x: x[0] / (np.sqrt(2 * math.pi) * x[2]) - x[3] / (np.sqrt(2 * math.pi) * x[5])},
        {"type": "ineq", "fun": lambda x: 5 * x[4] - x[1]},
        #        {'type': 'ineq', 'fun': lambda x: x[4] - x[1]-0.1},
    )
    if _cons is not None:
        cons = _cons

    init_params_ = [0 for i in range(len(init_params))]
    init_params_[0] = init_params[0]
    init_params_[1] = nm2ev(init_params[1])
    init_params_[2] = nm2ev(init_params[1]) - nm2ev(init_params[1] + init_params[2])
    init_params_[3] = init_params[3]
    init_params_[4] = nm2ev(init_params[4])
    init_params_[5] = nm2ev(init_params[4]) - nm2ev(init_params[4] + init_params[5])

    #     print(init_params_)

    result = minimize(gauss_func2, init_params_, args=(xdata, ydata), method="SLSQP", bounds=bounds, constraints=cons)
    #     result = minimize(gauss_func2, init_params_, args=(xdata,ydata), method='trust-constr', bounds=bounds, constraints=cons)
    """
    bounds: L-BFGS-B, TNC, SLSQP, Powell, and trust-constr
    constraints: COBYLA, SLSQP and trust-constr
    """
    a1, mew1, sigma1 = result.x[0], result.x[1], result.x[2]
    a2, mew2, sigma2 = result.x[3], result.x[4], result.x[5]

    if info is not None:
        info["a1"], info["mew1"], info["sigma1"] = result.x[0], result.x[1], result.x[2]
        info["intensity1"], info["fwhm1"] = (
            info["a1"] / (np.sqrt(2 * math.pi) * info["sigma1"]),
            2 * math.sqrt(2 * math.log(2)) * info["sigma1"],
        )
        info["a2"], info["mew2"], info["sigma2"] = result.x[3], result.x[4], result.x[5]
        info["intensity2"], info["fwhm2"] = (
            info["a2"] / (np.sqrt(2 * math.pi) * info["sigma2"]),
            2 * math.sqrt(2 * math.log(2)) * info["sigma2"],
        )
        info["result"] = result

    xdata_nm = np.array(se.index)
    ypred1 = (a1 / (np.sqrt(2 * math.pi) * sigma1)) * np.exp(-((nm2ev(xdata_nm) - mew1) ** 2) / (2 * sigma1**2))
    ypred2 = (a2 / (np.sqrt(2 * math.pi) * sigma2)) * np.exp(-((nm2ev(xdata_nm) - mew2) ** 2) / (2 * sigma2**2))
    ypred_sum = ypred1 + ypred2

    yred = np.array([ypred1, ypred2, ypred_sum]).transpose()
    # print(yred.shape)

    return pd.DataFrame(yred, index=se.index, columns=["g1", "g2", "gauss"])


def gauss_func3(parameter, x, y):
    a1, mew1, sigma1 = parameter[0], parameter[1], parameter[2]
    a2, mew2, sigma2 = parameter[3], parameter[4], parameter[5]
    a3, mew3, sigma3 = parameter[6], parameter[7], parameter[8]

    residual = (
        y
        - (a1 / (np.sqrt(2 * math.pi) * sigma1)) * np.exp(-((x - mew1) ** 2) / (2 * sigma1**2))
        - (a2 / (np.sqrt(2 * math.pi) * sigma2)) * np.exp(-((x - mew2) ** 2) / (2 * sigma2**2))
        - (a3 / (np.sqrt(2 * math.pi) * sigma3)) * np.exp(-((x - mew3) ** 2) / (2 * sigma3**2))
    )

    return np.linalg.norm(residual, ord=2)


# def uvdata_gauss3_fit_nm(se, _range=[400.5000, 450.0000], init_params=[1, 1, 5, 0.1, 0, 5, 0.1, 2, 5], info=None):
def uvdata_gauss3_fit_nm(se, _range, init_params, info=None):
    """[not used]uvdata gauss fit (in nm)"""
    xdata = np.array(se.loc[_range[0] : _range[1]].index)
    ydata = np.array(se.loc[_range[0] : _range[1]].values)

    bounds = []
    for i in range(len(init_params)):
        if i % 3 == 2:
            bounds.append((4, None))
        else:
            bounds.append((0, None))

    cons = (
        {"type": "ineq", "fun": lambda x: x[0] - x[3]},
        {"type": "ineq", "fun": lambda x: x[0] - x[6]},
        {"type": "ineq", "fun": lambda x: x[1] - x[4]},
        {"type": "ineq", "fun": lambda x: x[7] - x[1]},
    )

    result = minimize(gauss_func3, init_params, args=(xdata, ydata), method="SLSQP", bounds=bounds, constraints=cons)
    a1, mew1, sigma1 = result.x[0], result.x[1], result.x[2]
    a2, mew2, sigma2 = result.x[3], result.x[4], result.x[5]
    a3, mew3, sigma3 = result.x[6], result.x[7], result.x[8]

    if info is not None:
        info["a1"], info["mew1"], info["sigma1"] = result.x[0], result.x[1], result.x[2]
        info["intensity1"], info["fwhm1"] = (
            info["a1"] / (np.sqrt(2 * math.pi) * info["sigma1"]),
            2 * math.sqrt(2 * math.log(2)) * info["sigma1"],
        )
        info["fwhmev1"] = nm2ev(info["mew1"] - 0.5 * info["fwhm1"]) - nm2ev(info["mew1"] + 0.5 * info["fwhm1"])
        info["a2"], info["mew2"], info["sigma2"] = result.x[3], result.x[4], result.x[5]
        info["intensity2"], info["fwhm2"] = (
            info["a2"] / (np.sqrt(2 * math.pi) * info["sigma2"]),
            2 * math.sqrt(2 * math.log(2)) * info["sigma2"],
        )
        info["fwhmev2"] = nm2ev(info["mew2"] - 0.5 * info["fwhm2"]) - nm2ev(info["mew2"] + 0.5 * info["fwhm2"])
        info["a3"], info["mew3"], info["sigma3"] = result.x[6], result.x[7], result.x[8]
        info["intensity3"], info["fwhm3"] = (
            info["a3"] / (np.sqrt(2 * math.pi) * info["sigma3"]),
            2 * math.sqrt(2 * math.log(2)) * info["sigma3"],
        )
        info["fwhmev3"] = nm2ev(info["mew3"] - 0.5 * info["fwhm3"]) - nm2ev(info["mew3"] + 0.5 * info["fwhm3"])

    xdata = np.array(se.index)
    ypred1 = (a1 / (np.sqrt(2 * math.pi) * sigma1)) * np.exp(-((xdata - mew1) ** 2) / (2 * sigma1**2))
    ypred2 = (a2 / (np.sqrt(2 * math.pi) * sigma2)) * np.exp(-((xdata - mew2) ** 2) / (2 * sigma2**2))
    ypred3 = (a3 / (np.sqrt(2 * math.pi) * sigma3)) * np.exp(-((xdata - mew3) ** 2) / (2 * sigma3**2))

    yred = np.array([ypred1, ypred2, ypred3]).transpose()
    # print(yred.shape)

    return pd.DataFrame(yred, index=se.index, columns=["g1", "g2", "g3"])


def uvdata_gauss3_fit(se, _range, init_params, info=None):
    # def uvdata_gauss3_fit(se, _range=[400.5000, 450.0000], init_params=[1, 1, 5, 0.1, 2, 5, 0.1, 0, 5], info=None):
    """uvdata gauss fit (in eV)
    init_params: [a1,mew1,sigma1,...]
    """
    xdata_ = nm2ev(np.array(se.loc[_range[0] : _range[1]].index))[::-1]
    ydata_ = np.array(se.loc[_range[0] : _range[1]].values)[::-1]

    xdata = xdata_
    ydata = ydata_

    bounds = []
    for i in range(len(init_params)):
        if i == 1:
            bounds.append((nm2ev(_range[1]), nm2ev(_range[0] + 20)))
        elif i % 3 == 1:
            bounds.append((nm2ev(_range[1]), nm2ev(_range[0])))
        elif i == 2:
            #             bounds.append((sigma2fwhm(0.07), sigma2fwhm(0.075)))
            bounds.append((sigma2fwhm(0.07), sigma2fwhm(0.09)))
        #             bounds.append((sigma2fwhm(0.07), sigma2fwhm(0.09)))
        #             bounds.append((0.01,0.09/(2*math.sqrt(2*math.log(2)))))
        elif i % 3 == 2:
            bounds.append((sigma2fwhm(0.07), sigma2fwhm(0.2)))
        else:
            bounds.append((0, None))

    cons = (
        {"type": "ineq", "fun": lambda x: x[0] / (np.sqrt(2 * math.pi) * x[2]) - x[3] / (np.sqrt(2 * math.pi) * x[5])},
        {"type": "ineq", "fun": lambda x: x[0] / (np.sqrt(2 * math.pi) * x[2]) - x[6] / (np.sqrt(2 * math.pi) * x[8])},
        {"type": "ineq", "fun": lambda x: x[4] - x[1] - 0.1},
        {"type": "ineq", "fun": lambda x: x[1] - x[7] - 0.1},
    )

    init_params_ = [0 for i in range(len(init_params))]
    init_params_[0] = init_params[0]
    init_params_[1] = nm2ev(init_params[1])
    init_params_[2] = nm2ev(init_params[1]) - nm2ev(init_params[1] + init_params[2])
    init_params_[3] = init_params[3]
    init_params_[4] = nm2ev(init_params[4])
    init_params_[5] = nm2ev(init_params[4]) - nm2ev(init_params[4] + init_params[5])
    init_params_[6] = init_params[6]
    init_params_[7] = nm2ev(init_params[7])
    init_params_[8] = nm2ev(init_params[7]) - nm2ev(init_params[7] + init_params[8])

    #     print(init_params_)

    result = minimize(gauss_func3, init_params_, args=(xdata, ydata), method="SLSQP", bounds=bounds, constraints=cons)
    #     result = minimize(gauss_func3, init_params_, args=(xdata,ydata), method='trust-constr', bounds=bounds, constraints=cons)
    """
    bounds: L-BFGS-B, TNC, SLSQP, Powell, and trust-constr
    constraints: COBYLA, SLSQP and trust-constr
    """
    a1, mew1, sigma1 = result.x[0], result.x[1], result.x[2]
    a2, mew2, sigma2 = result.x[3], result.x[4], result.x[5]
    a3, mew3, sigma3 = result.x[6], result.x[7], result.x[8]

    if info is not None:
        info["a1"], info["mew1"], info["sigma1"] = result.x[0], result.x[1], result.x[2]
        info["intensity1"], info["fwhm1"] = (
            info["a1"] / (np.sqrt(2 * math.pi) * info["sigma1"]),
            2 * math.sqrt(2 * math.log(2)) * info["sigma1"],
        )
        info["a2"], info["mew2"], info["sigma2"] = result.x[3], result.x[4], result.x[5]
        info["intensity2"], info["fwhm2"] = (
            info["a2"] / (np.sqrt(2 * math.pi) * info["sigma2"]),
            2 * math.sqrt(2 * math.log(2)) * info["sigma2"],
        )
        info["a3"], info["mew3"], info["sigma3"] = result.x[6], result.x[7], result.x[8]
        info["intensity3"], info["fwhm3"] = (
            info["a3"] / (np.sqrt(2 * math.pi) * info["sigma3"]),
            2 * math.sqrt(2 * math.log(2)) * info["sigma3"],
        )

    xdata_nm = np.array(se.index)
    ypred1 = (a1 / (np.sqrt(2 * math.pi) * sigma1)) * np.exp(-((nm2ev(xdata_nm) - mew1) ** 2) / (2 * sigma1**2))
    ypred2 = (a2 / (np.sqrt(2 * math.pi) * sigma2)) * np.exp(-((nm2ev(xdata_nm) - mew2) ** 2) / (2 * sigma2**2))
    ypred3 = (a3 / (np.sqrt(2 * math.pi) * sigma3)) * np.exp(-((nm2ev(xdata_nm) - mew3) ** 2) / (2 * sigma3**2))
    ypred_sum = ypred1 + ypred2 + ypred3

    yred = np.array([ypred1, ypred2, ypred3, ypred_sum]).transpose()
    # print(yred.shape)

    return pd.DataFrame(yred, index=se.index, columns=["g1", "g2", "g3", "gauss"])


def uvdata_check_class(
    se,
    _ranges,
    peaks_thres,
    mol_name="TPP",
    slv_name="any",
    # _ranges=[[480.0, 540.0], [380.0, 440.0]],
    _width=20.0,
    # peaks_thres=[0.0005, 0.01],
    large_peak_thres_ratio=20,
    info=None,
    nclass=3,
    verbose=0,
):
    """[obsolete]uvdata check spectrum class from peaks
    Returns:
       int: 0 (insoluble), 1 (overflow/scattering), 2 (soluble)
       int: 0 (insoluble), 1 (soluble) ... nclass=2
    """
    ret = -1
    ignore_large_value = 7.0
    vec = se
    # vec = uvdata_smooth(se, _range=_ranges[0], n=10)
    # vec = uvdata_smooth(vec, _ranges[1], n=10)

    # detect peaks in 1st range
    peaks = detect_peaks(vec, _range=_ranges[0], _width=_width, peak_thres=peaks_thres[0], verbose=verbose)

    _is_large_peak = False
    if 0 < len(peaks) and peaks_thres[0] * large_peak_thres_ratio < max(peaks, key=(lambda x: x[1]))[1]:
        _is_large_peak = True

    # detect peaks in 2nd range
    peaks_2nd = detect_peaks(vec, _range=_ranges[1], _width=_width, peak_thres=peaks_thres[1], verbose=verbose)

    if 0 < verbose:
        print(mol_name, slv_name, peaks + peaks_2nd)

    if nclass == 3:
        """ret: 0 (insoluble) | 1 (overflow/scattering) | 2 (measured)"""
        if mol_name[0:3] != "DPP" and ignore_large_value < se.loc[398.000:402.0000].max():
            """too large at 400.000 => overflown"""
            if info is not None:
                info["peaks"] = peaks
            ret = 1
        elif len(peaks) < 1 or (0 < len(peaks) and len(peaks_2nd) < 1 and not _is_large_peak):
            """too large at 400.000 => overflown"""
            if info is not None:
                info["peaks"] = peaks + peaks_2nd
            ret = 0
        else:
            if info is not None:
                info["peaks"] = peaks + peaks_2nd
            ret = 2

    elif nclass == 2:
        """ret: 0 (insoluble) | 1 (soluble)"""
        if mol_name[0:3] != "DPP" and ignore_large_value < se.loc[398.000:402.0000].max() and 0 < len(peaks):
            if info is not None:
                info["peaks"] = peaks
            ret = 1
        elif (mol_name[0:3] != "DPP" and ignore_large_value < se.loc[398.000:402.0000].max()) or (
            len(peaks) < 1 or (0 < len(peaks) and len(peaks_2nd) < 1 and not _is_large_peak)
        ):
            if info is not None:
                info["peaks"] = peaks + peaks_2nd
            ret = 0
        else:
            if info is not None:
                info["peaks"] = peaks + peaks_2nd
            ret = 1

    else:
        raise RuntimeError("nclass must be 3 or 2:" + nclass)

    return ret


def uvdata_check_class2(
    se,
    _ranges,
    peaks_thres,
    mol_name="TPP",
    slv_name="any",
    # _ranges=[[480.0, 540.0], [380.0, 440.0]],
    _width=20.0,
    # peaks_thres=[0.0005, 0.01],
    large_peak_thres_ratio=20,
    info=None,
    nclass=3,
    verbose=0,
):
    """uvdata check spectrum class from peaks
    Returns:
       int: 0 (insoluble), 1 (overflow/scattering), 2 (soluble)
       int: 0 (insoluble), 1 (soluble) ... nclass=2
    """
    ret = -1
    ignore_large_value = 7.0
    vec = se
    # vec = uvdata_smooth(se, _range=_ranges[0], n=10)
    # vec = uvdata_smooth(vec, _ranges[1], n=10)

    # detect peaks in 1st range
    peaks = detect_peaks(vec, _range=_ranges[0], _width=_width, peak_thres=peaks_thres[0], verbose=verbose)

    _is_large_peak = False
    if 0 < len(peaks) and peaks_thres[0] * large_peak_thres_ratio < max(peaks, key=(lambda x: x[1]))[1]:
        _is_large_peak = True

    # detect peaks in 2nd range
    peaks_2nd = detect_peaks(vec, _range=_ranges[1], _width=_width, peak_thres=peaks_thres[1], verbose=verbose)

    # remove mis-detected peaks at 400.0 or 400.5 discontinuous point
    if _ranges[1][0] <= 400.0 and 400.0 < _ranges[1][1]:
        idx_to_remove = []
        for i, p in enumerate(peaks_2nd):
            if 400.0 <= p[0] and p[0] <= 400.5 and 10 * abs(se.loc[400.0] - se.loc[399.5]) < abs(se.loc[400.5] - se.loc[400.0]):
                idx_to_remove.append(i)

        for i in sorted(idx_to_remove, reverse=True):
            peaks_2nd.pop(i)

    _is_large_peak_2nd = False
    if 0 < len(peaks_2nd) and peaks_thres[1] * large_peak_thres_ratio < max(peaks_2nd, key=(lambda x: x[1]))[1]:
        _is_large_peak_2nd = True

    if 0 < verbose:
        print(mol_name, slv_name, peaks + peaks_2nd)

    if nclass == 3:
        """ret: 0 (insoluble) | 1 (overflow/scattering) | 2 (measured)"""
        if mol_name[0:3] != "DPP" and ignore_large_value < se.loc[398.000:402.0000].max():
            """too large at 400.000 => overflown"""
            if info is not None:
                info["peaks"] = peaks
            ret = 1
        elif (len(peaks) < 1 and len(peaks_2nd) < 1) or (0 < len(peaks) and len(peaks_2nd) < 1 and not _is_large_peak):
            """no peaks or only small peak"""
            if info is not None:
                info["peaks"] = peaks + peaks_2nd
            ret = 0
        elif len(peaks) < 1 and 0 < len(peaks_2nd) and _is_large_peak_2nd:
            """2nd peaks (420nm) only"""
            if info is not None:
                info["peaks"] = peaks + peaks_2nd
            ret = 1
        else:
            if info is not None:
                info["peaks"] = peaks + peaks_2nd
            ret = 2

    elif nclass == 2:
        """ret: 0 (insoluble) | 1 (soluble)"""
        if mol_name[0:3] != "DPP" and ignore_large_value < se.loc[398.000:402.0000].max() and 0 < len(peaks):
            if info is not None:
                info["peaks"] = peaks
            ret = 1
        elif (mol_name[0:3] != "DPP" and ignore_large_value < se.loc[398.000:402.0000].max()) or (
            (len(peaks) < 1 and len(peaks_2nd) < 1) or (0 < len(peaks) and len(peaks_2nd) < 1 and not _is_large_peak)
        ):
            if info is not None:
                info["peaks"] = peaks + peaks_2nd
            ret = 0
        else:
            if info is not None:
                info["peaks"] = peaks + peaks_2nd
            ret = 1

    else:
        raise RuntimeError("nclass must be 3 or 2:" + nclass)

    return ret
