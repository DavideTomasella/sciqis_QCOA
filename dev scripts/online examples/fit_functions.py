from typing import Dict, Tuple, Callable
import numpy as np
from scipy import optimize
from scipy import signal
from utils import *
from numpy.ma import average
import lmfit
from numba import njit

# # Fit Functions
# Fit functions based on Kharel et al.'s  model (normalized intensity).

#
#  |                               k2/2                   | ^ 2
#  |              --------------------------------------- |
#  | 1   -   A *                           N1 * g0^2      |
#  |              i(w-w0+D2) - k2/2 + ------------------- |
#  |                                  -i(w-w0) - Gamma/2  |
#
#     A : normalization factor
#     w : frequency [GHz]
#    w0 : resonance frequency [GHz] 12.45GHz
#         w-w0 === δ = Ω − Ωm = ωp − ωl − Ωa
#    k2 : optical decay rate [MHz] 1MHz
# Gamma : acoustic decay rate [MHz] 10MHz
#    D2 : detuning [MHz] 0MHz
#         ∆2 = Ωa + ωl − ω2
#    N1 : number pumping photons [1]
#    g0 : single photon coupling strength [Hz] 1Hz
#


def get_fit_function(name: str) -> Callable:
    if name.lower().endswith("detuning") or name.lower().endswith("mostfreeparams"):
        return Kharel_normalized_model
    elif name.lower().endswith("optical"):
        return optical_line
    else:
        return Kharel_normalized_model


def Kharel_model(w, **args):
    P = MHz_to_rad_s(args["k2"] / 2) / (
        (
            1j * (GHz_to_rad_s(w - args["w0"]) + MHz_to_rad_s(args["D2"]))
            - MHz_to_rad_s(args["k2"] / 2)
            + (
                -Hz_to_rad_s(args["g0"]) ** 2  # TODO DT: add - for stokes
                * args["N1"]
                / (1j * GHz_to_rad_s(w - args["w0"]) - MHz_to_rad_s(args["Gamma"] / 2))
            )
        )
    )
    return P


def Kharel_normalized_model(d, **args):
    P = np.minimum(2.3, np.abs(1 - args["A"] * Kharel_model(d, **args)) ** 2)
    return P


@njit(parallel=True)
def Kharel_model_array(w, pars):
    # {"A": 0, "D2": 1, "k2": 2, "g0": 3,"Gamma": 4, "N1": 5, "w0": 6}
    P = MHz_to_rad_s(pars[2] / 2) / (
        (
            1j * (GHz_to_rad_s(w - pars[6]) + MHz_to_rad_s(pars[1]))
            - MHz_to_rad_s(pars[2] / 2)
            + (
                Hz_to_rad_s(pars[3]) ** 2
                * pars[5]
                / (1j * GHz_to_rad_s(w - pars[6]) - MHz_to_rad_s(pars[4] / 2))
            )
        )
    )
    return P


@njit(parallel=True)
def Kharel_normalized_model_array(d, pars):
    P = np.minimum(2.5, np.abs(1 - pars[0] * Kharel_model_array(d, pars)) ** 2)
    return P


def optical_line(w, **args):
    P = (
        np.abs(
            1
            - args["A"]
            * MHz_to_rad_s(args["k2"] / 2)
            / (1j * (GHz_to_rad_s(w - args["w0"])) - MHz_to_rad_s(args["k2"] / 2))
        )
        ** 2
    )
    return P


# Parametrized fitting function
# Through fit_pars_tunable_mask, we can choose which parameters to fit
# and which to fix to their initial values fit_pars_zero


def fit_parametric_function(x, fit_fun: Callable, fit_pars: Dict, fit_config: Dict):
    pars = fit_config["zero"].copy()
    for name, value in fit_pars.items():
        if fit_config["level"][name] >= 0:
            pars[name] = value
    return fit_fun(x, **pars)


def init_fit_pars(data: np.ndarray, fit_config: Dict, ignore_edges: int):
    fit_pars_zero = fit_config["zero"]
    fit_pars_bounds = fit_config["bounds"]
    fit_pars_tunable_mask = fit_config["level"]
    # Refining guess of D2_0 if it is tunable
    # for D2 I want to find the position of the minimum value(ignoring the edge artifacts),
    # and take the difference between that and w0
    if fit_pars_tunable_mask["D2"] >= 0:
        min_index = np.argmin(data[ignore_edges:-ignore_edges, 1]) + ignore_edges
        min_freq = data[min_index, 0]
        fit_pars_zero["D2"] = GHz_to_MHz(fit_pars_zero["w0"] - min_freq)
    zeros = [
        fit_pars_zero[key]
        for key in fit_pars_tunable_mask
        if fit_pars_tunable_mask[key] >= 0
    ]
    lbound = [
        fit_pars_bounds[0][key]
        for key in fit_pars_tunable_mask
        if fit_pars_tunable_mask[key] >= 0
    ]
    hbound = [
        fit_pars_bounds[1][key]
        for key in fit_pars_tunable_mask
        if fit_pars_tunable_mask[key] >= 0
    ]
    return zeros, lbound, hbound


def init_lmfit_pars(
    data_list: Union[List[np.ndarray], List[List[np.ndarray]]],
    fit_config: Dict,
    init_coupling: List[float] = None,
    ignore_edges: int = 0,
):
    # Init Lm_Fit parameters structure
    pars = lmfit.Parameters()
    sep: str = "___"
    max_level = max(fit_config["level"].values())
    split_index = np.cumsum([0] + [len(data) for data in data_list])

    # for data_i, data in enumerate(flatten(data_list)):
    data_i = -1
    for level, data_level in enumerate(data_list):
        if init_coupling is not None:
            # fix the following code that gives the error: 'int' object is not subscriptable

            fit_config["zero"]["N1"] = (
                init_coupling[level] ** 2 / fit_config["zero"]["g0"]
            )

        for data in data_level:
            data_i += 1
            # New init values for each data set
            # Detuning / resonance frequency guess is based on the data peaks
            peak_id, peaks = signal.find_peaks(
                1 - data[ignore_edges:-ignore_edges, 1], prominence=0.06
            )
            if np.max(data[ignore_edges:-ignore_edges, 1]) > 1.03:
                peak_id2, peaks2 = signal.find_peaks(
                    data[ignore_edges:-ignore_edges, 1] - 1, prominence=0.03
                )
                peak_id = np.append(peak_id, peak_id2)
                peaks["prominences"] = np.append(
                    peaks["prominences"], peaks2["prominences"]
                )
            if len(peak_id) > 0:
                min_freq = np.average(
                    data[peak_id + ignore_edges, 0],
                    weights=np.sqrt(peaks["prominences"]),
                )
                if fit_config["level"]["D2"] == 0:
                    fit_config["zero"]["D2"] = GHz_to_MHz(
                        fit_config["zero"]["w0"] - min_freq
                    )
                    # * np.sum(peaks["prominences"])
                    # range = [0.9 * fit_config["zero"]["D2"], 1.1 * fit_config["zero"]["D2"]]
                    # fit_config["bounds"][0]["D2"] = min(range)
                    # fit_config["bounds"][1]["D2"] = max(range)
                elif fit_config["level"]["w0"] == 0:
                    fit_config["zero"]["w0"] = min_freq
            # Initialize parameters
            for par_name in fit_config["zero"].keys():
                if fit_config["level"][par_name] >= 0:
                    # Add parameters to the list
                    pars.add(
                        f"p{data_i}{sep}{par_name}",
                        value=fit_config["zero"][par_name],
                        min=fit_config["bounds"][0][par_name],
                        max=fit_config["bounds"][1][par_name],
                    )
                    # Duplicate values when the parameter is common among the datasets
                    if (
                        max_level > 0
                        and data_i > 0
                        and fit_config["level"][par_name] == max_level
                    ):
                        pars[f"p{data_i}{sep}{par_name}"].expr = f"p{0}{sep}{par_name}"
                    # (data_i==split_index).any():
                    if (
                        max_level > 1
                        and not data_i in split_index
                        and fit_config["level"][par_name] == max_level - 1
                    ):
                        pars[
                            f"p{data_i}{sep}{par_name}"
                        ].expr = f"p{max([i for i in split_index if i < data_i])}{sep}{par_name}"

    return pars


def fit_sigle_data(
    fit_function, data: np.ndarray, ignore_edges: int, zeros: List, bounds: Tuple[List]
):
    pars, parcov = optimize.curve_fit(
        fit_function,
        data[ignore_edges:-ignore_edges, 0],
        data[ignore_edges:-ignore_edges, 1],
        p0=zeros,
        bounds=bounds,
    )
    errors = np.array([np.sqrt(parcov[i][i]) for i in range(len(pars))])
    return pars, errors, parcov


def evaluate_curve_fit(
    select_fit: str,
    data_list: Union[List[np.ndarray], List[List[np.ndarray]]],
    fit_config: Dict,
    init_coupling: List[float] = None,
    ignore_edges: str = 0,
):
    """Fits the data, with the optical linewith and g0 fixed. Returns an array [A, D2, Gamma, N1, w0]"""
    # Define the fitting model
    inner_fit_function = get_fit_function(select_fit)
    # Initialize output structures
    fit_res: Dict[str, np.ndarray] = {}
    fit_err: Dict[str, np.ndarray] = {}
    for key in fit_config["zero"].keys():
        fit_res[key] = np.asarray(
            [fit_config["zero"][key]] * len(flatten(data_list)), dtype=np.float32
        )
        fit_err[key] = np.asarray(
            [fit_config["zero"][key]] * len(flatten(data_list)), dtype=np.float32
        )

    if select_fit.lower().startswith("lmfit"):
        # Define the minimization function for LmFit
        # @njit(parallel=True)
        def objective(
            params,
            x: np.ndarray,
            x0: np.ndarray,
            data: np.ndarray,
            fun_pars: Dict,
            tunable: Dict,
        ) -> np.ndarray:
            ndata, _ = data.shape
            resid = np.zeros_like(data)
            sep: str = "___"
            if True:
                for i in range(ndata):
                    for par in fun_pars.keys():
                        if tunable[par] >= 0:
                            fun_pars[par] = params[f"p{i}{sep}{par}"]
                    resid[i, :] = data[i, :] - inner_fit_function(x + x0[i], **fun_pars)
            else:
                for i in range(ndata):
                    for par in fun_pars.keys():
                        if tunable[par] >= 0:
                            fun_pars[par] = params[f"p{i}{sep}{par}"]
                    resid[i, :] = data[i, :] - Kharel_normalized_model_array(
                        x, fun_pars.values()
                    )
            return resid.flatten()

        # Initialize lmfit parameters
        lmfit_pars = init_lmfit_pars(data_list, fit_config, init_coupling, ignore_edges)
        # Create nd matrix with all the dataset
        # NOTE: we assume all the dataset having the same frequency step and length BUT they could have different starting frequency
        start_freq = np.array([data[ignore_edges, 0] for data in flatten(data_list)])
        freq = flatten(data_list)[0][ignore_edges:-ignore_edges, 0] - start_freq[0]
        values = np.array(
            [data[ignore_edges:-ignore_edges, 1] for data in flatten(data_list)]
        )
        fun_pars = fit_config["zero"].copy()

        lmfit_result = lmfit.minimize(
            objective,
            lmfit_pars,
            args=(freq, start_freq, values, fun_pars, fit_config["level"]),
            method="leastsq",
            nan_policy="omit",
        )
        lmfit.report_fit(lmfit_result, show_correl=False)
        sep = "___"
        for data_id in range(len(flatten(data_list))):
            for idx, key in enumerate(fit_config["zero"].keys()):
                if fit_config["level"][key] >= 0:
                    fit_res[key][data_id] = lmfit_result.params[
                        f"p{data_id}{sep}{key}"
                    ].value
                    fit_err[key][data_id] = lmfit_result.params[
                        f"p{data_id}{sep}{key}"
                    ].stderr
    else:
        # Define the fitting function for single_data fitting
        if select_fit.lower() == "detuning":

            def fit_function(x, A, D2):
                pars = {"A": A, "D2": D2}
                return fit_parametric_function(x, inner_fit_function, pars, fit_config)

        elif select_fit.lower() == "mostfreeparams":

            def fit_function(x, A, D2, Gamma, N1, w0):
                pars = {"A": A, "D2": D2, "Gamma": Gamma, "N1": N1, "w0": w0}
                return fit_parametric_function(x, inner_fit_function, pars, fit_config)

        elif select_fit.lower() == "optical":

            def fit_function(x, A, k2, w0):
                pars = {"A": A, "k2": k2, "w0": w0}
                return fit_parametric_function(x, inner_fit_function, pars, fit_config)

        else:

            def fit_function(x, A, D2, Gamma, N1, w0):
                pars = {"A": A, "D2": D2, "Gamma": Gamma, "N1": N1, "w0": w0}
                return fit_parametric_function(
                    x, Kharel_normalized_model, pars, fit_config
                )

            print("Invalid fit function name. Using Kharel fit function.")

        for i, data in enumerate(flatten(data_list)):
            zeros, lbound, hbound = init_fit_pars(data, fit_config, ignore_edges)
            # fit the data
            try:
                parameters, errors, covariance = fit_sigle_data(
                    fit_function, data, ignore_edges, zeros, (lbound, hbound)
                )
                # organize the results keeping the fixed parameters
                for idx, key in enumerate(fit_config["zero"].keys()):
                    if fit_config["level"][key] >= 0:
                        fit_res[key][i] = parameters[idx]
                        fit_err[key][i] = errors[idx]
            except (
                RuntimeError
            ) as e:  # If the fitting fails, it just prints an error message and continues
                print(" Fit failed for '{}'".format(i))
                print(e)

        # Recover data structure for the fitting results
    if max(fit_config["level"].values()) > 1:
        split_index = np.cumsum([len(data) for data in data_list])
        for key in fit_config["zero"].keys():
            fit_res[key] = np.split(fit_res[key], split_index)
            fit_err[key] = np.split(fit_err[key], split_index)

    return fit_res, fit_err


# # Fit Results


def get_i_params(fit_params: Dict, i: int):
    params = {}
    for par, value in fit_params.items():
        params[par] = value[i]
    return params


def get_grp_i_params(fit_params: Dict, grp: int, i: int):
    params = {}
    for par, value in fit_params.items():
        params[par] = value[grp][i]
    return params


def calc_fit_averages(
    fit_res: Dict[str, np.ndarray],
    fit_err: Dict[str, np.ndarray],
    fit_common: Dict[str, bool],
):
    """Calculates the average of the fit results, using the fit errors as weights."""
    fit_mean, fit_wmean, fit_geom, fit_wgeom, fit_std = {}, {}, {}, {}, {}
    for key in fit_res.keys():
        outliers = is_outlier(fit_res[key])
        fit_res_nooutliers = fit_res[key][~outliers]
        fit_err_nooutliers = fit_err[key][~outliers]
        fit_mean[key] = average(fit_res_nooutliers)
        fit_wmean[key] = average(
            fit_res_nooutliers, weights=1 / np.power(fit_err_nooutliers, 2)
        )
        fit_wgeom[key] = np.exp(
            average(
                np.log(np.abs(fit_res_nooutliers)),
                weights=1 / np.power(fit_err_nooutliers, 2),
            )
        )
        fit_geom[key] = np.exp(average(np.log(np.abs(fit_res_nooutliers))))
        fit_std[key] = np.sqrt(average(np.power(fit_err_nooutliers, 2)))
        if fit_common[key] == 0:  # TODO
            fit_std[key] = fit_std[key] / np.sqrt(len(fit_err_nooutliers))
    return fit_mean, fit_wmean, fit_geom, fit_wgeom, fit_std


def calc_fit_group_averages(
    fit_res: Dict[str, np.ndarray],
    fit_err: Dict[str, np.ndarray],
    grp: int,
    fit_common: Dict[str, bool],
):
    """Calculates the average of the fit results, using the fit errors as weights."""
    fit_mean, fit_wmean, fit_geom, fit_wgeom, fit_std = {}, {}, {}, {}, {}
    for key in fit_res.keys():
        outliers = is_outlier(fit_res[key][grp])
        fit_res_nooutliers = fit_res[key][grp][~outliers]
        fit_err_nooutliers = fit_err[key][grp][~outliers]
        fit_mean[key] = average(fit_res_nooutliers)
        fit_wmean[key] = average(
            fit_res_nooutliers, weights=1 / np.power(fit_err_nooutliers, 2)
        )
        fit_wgeom[key] = np.exp(
            average(
                np.log(np.abs(fit_res_nooutliers)),
                weights=1 / np.power(fit_err_nooutliers, 2),
            )
        )
        fit_geom[key] = np.exp(average(np.log(np.abs(fit_res_nooutliers))))
        fit_std[key] = np.sqrt(average(np.power(fit_err_nooutliers, 2)))
        if fit_common[key] == 0:  # TODO
            fit_std[key] = fit_std[key] / np.sqrt(len(fit_err_nooutliers))
    return fit_mean, fit_wmean, fit_geom, fit_wgeom, fit_std
