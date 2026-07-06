import os
import glob
import shutil
import numpy as np
import pandas as pd
from pcmdi_metrics.graphics import (
    Metrics,
    normalize_by_median,
    parallel_coordinate_plot,
    portrait_plot
)

# ============================================================
# Helper Functions from mean_climate_plot_parser
# ============================================================
def metrics_inquire(name):
    metrics = {
        "std-obs_xy": "Spatial Standard Deviation (Reference)",
        "std_xy": "Spatial Standard Deviation (Model)",
        "std-obs_xyt": "Spatial-temporal Standard Deviation (Reference)",
        "std_xyt": "Spatial-temporal Standard Deviation (Model)",
        "std-obs_xy_devzm": "Standard Deviation of Deviation from Zonal Mean (Reference)",
        "mean_xy": "Area Weighted Spatial Mean (Model)",
        "mean-obs_xy": "Area Weighted Spatial Mean (Reference)",
        "std_xy_devzm": "Standard Deviation of Deviation from Zonal Mean (Model)",
        "rms_xyt": "Spatio-Temporal Root Mean Square Error",
        "rms_xy": "Spatial Root Mean Square Error",
        "rmsc_xy": "Centered Spatial Root Mean Square Error",
        "cor_xy": "Spatial Pattern Correlation Coefficient",
        "bias_xy": "Mean Bias (Model - Reference)",
        "mae_xy": "Mean Absolute Difference (Model - Reference)",
        "rms_y": "Root Mean Square Error of Zonal Mean",
        "rms_devzm": "Root Mean Square Error of Deviation From Zonal Mean",
    }
    return metrics.get(name, name)

def find_latest(pmprdir, mip, exp):
    versions = sorted(
        [r.split("/")[-1] for r in glob.glob(os.path.join(pmprdir, mip, exp, "v????????"))]
    )
    return versions[-1]

def shift_row_to_bottom(df, index_to_shift):
    idx = [i for i in df.index if i != index_to_shift]
    return df.loc[idx + [index_to_shift]]

def find_cmip_metric_data(pmprdir, data_set, var):
    mip = data_set.split(".")[0]
    exp = data_set.split(".")[1]
    case_id = data_set.split(".")[2]
    if case_id == "":
        case_id = find_latest(pmprdir, mip, exp)
    fpath = glob.glob(os.path.join(pmprdir, mip, exp, case_id, "{}.*.json".format(var)))
    if len(fpath) < 1 and var == "rtmt":
        fpath = glob.glob(os.path.join(pmprdir, mip, exp, case_id, "{}.*.json".format("rt")))
    if len(fpath) > 0 and os.path.exists(fpath[0]):
        return fpath[0], 0
    else:
        print("Warning: cmip metrics data not found for {}....".format(var))
        return None, -99

def select_models(df, selected_models):
    model_names = df["model"].tolist()
    for model_name in model_names:
        drop_model = True
        for keyword in selected_models:
            if keyword in model_name:
                drop_model = False
                break
        if drop_model:
            df.drop(df.loc[df["model"] == model_name].index, inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df

def exclude_models(df, excluded_models):
    model_names = df["model"].tolist()
    for model_name in model_names:
        drop_model = False
        for keyword in excluded_models:
            if keyword in model_name:
                drop_model = True
                break
        if drop_model:
            df.drop(df.loc[df["model"] == model_name].index, inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df

def fill_plot_var_and_units(model_lib, variables):
    units_all = {
        "prw": "[kg m$^{-2}$]",
        "pr": "[mm d$^{-1}$]",
        "prsn": "[mm d$^{-1}$]",
        "prc": "[mm d$^{-1}$]",
        "hfls": "[W m$^{-2}$]",
        "hfss": "[W m$^{-2}$]",
        "clivi": "[kg $m^{-2}$]",
        "clwvi": "[kg $m^{-2}$]",
        "psl": "[Pa]",
        "rlds": "[W m$^{-2}$]",
        "rldscs": "[W $m^{-2}$]",
        "rtmt": "[W m$^{-2}$]",
        "rlus": "[W m$^{-2}$]",
        "rluscs": "[W m$^{-2}$]",
        "rlut": "[W m$^{-2}$]",
        "rlutcs": "[W m$^{-2}$]",
        "rsds": "[W m$^{-2}$]",
        "rsdscs": "[W m$^{-2}$]",
        "rstcre": "[W m$^{-2}$]",
        "rltcre": "[W m$^{-2}$]",
        "rsus": "[W m$^{-2}$]",
        "rsuscs": "[W m$^{-2}$]",
        "ts": "[K]",
        "tas": "[K]",
        "tauu": "[Pa]",
        "tauv": "[Pa]",
        "zg500": "[m]",
        "ta200": "[K]",
        "ta850": "[K]",
        "ua200": "[m s$^{-1}$]",
        "ua850": "[m s$^{-1}$]",
        "va200": "[m s$^{-1}$]",
        "va850": "[m s$^{-1}$]",
        "tasmin": "[K]",
        "tasmax": "[K]",
        "clt": "[%]",
    }

    variable_units = []
    variable_names = []
    for var in variables:
        if var in units_all.keys():
            varunt = var + "\n" + str(units_all[var])
            variable_units.append(varunt)
            variable_names.append(var)
            for stat in model_lib:
                for season in model_lib[stat]:
                    for region in model_lib[stat][season]:
                        df = pd.DataFrame(model_lib[stat][season][region])
                        for i, model in enumerate(df["model"].tolist()):
                            if model in ['E3SM-1-0', 'E3SM-1-1-ECA']:
                                idxs = df[df.iloc[:, 0] == model].index
                                df.loc[idxs, "ta850"] = np.nan
                            if model in ['CIESM']:
                                idxs = df[df.iloc[:, 0] == model].index
                                df.loc[idxs, "pr"] = np.nan
                        model_lib[stat][season][region] = df
        else:
            print("Warning: {} is not found in metrics data".format(var))
            for stat in model_lib:
                for season in model_lib[stat]:
                    for region in model_lib[stat][season]:
                        df = pd.DataFrame(model_lib[stat][season][region])
                        if var in df.columns:
                            df = df.drop(columns=var)
                        model_lib[stat][season][region] = df
    return model_lib, variable_names, variable_units

def find_metrics_data(parameter):
    pmp_set = parameter.pcmdi_data_set
    pmp_path = parameter.pcmdi_data_path
    test_set = parameter.test_data_set
    test_path = parameter.test_data_path
    refr_set = parameter.refr_data_set
    refr_path = parameter.refr_data_path
    run_type = parameter.run_type
    debug = parameter.debug

    test_mip = test_set.split(".")[0]
    test_exp = test_set.split(".")[1]
    test_case_id = test_set.split(".")[-1]
    test_dir = os.path.join(test_path, test_mip, test_exp, test_case_id)
    if run_type == "model_vs_model":
        refr_mip = refr_set.split(".")[0]
        refr_exp = refr_set.split(".")[1]
        refr_case_id = refr_set.split(".")[-1]
        refr_dir = os.path.join(refr_path, refr_mip, refr_exp, refr_case_id)

    variables = [
        s.split("/")[-1].split("_")[0]
        for s in glob.glob(os.path.join(test_dir, "*{}.json".format(test_case_id)))
        if os.path.exists(s)
    ]

    test_list = []
    refr_list = []
    cmip_list = []

    for vv in variables:
        ftest = glob.glob(os.path.join(test_dir, "{}_*_{}.json".format(vv, test_case_id)))
        fcmip, rcode = find_cmip_metric_data(pmp_path, pmp_set, vv)
        if rcode == 0:
            if len(ftest) > 0 and fcmip and len(fcmip) > 0:
                test_list.append(ftest[0])
                cmip_list.append(fcmip)
            if run_type == "model_vs_model":
                frefr = glob.glob(os.path.join(refr_dir, "{}_*_{}.json".format(vv, refr_case_id)))
                if len(frefr) > 0:
                    refr_list.append(frefr[0])
        del (ftest, fcmip)
    return test_list, refr_list, cmip_list

# ============================================================
# Core Diagnostic Execution
# ============================================================
def load_test_model_data(test_file, refr_file, mip, run_type):
    pd.set_option("future.no_silent_downcasting", True)
    test_lib = Metrics(test_file)

    if run_type == "model_vs_model":
        refr_lib = Metrics(refr_file)
        test_lib = test_lib.merge(refr_lib)

    test_models = []
    for stat in test_lib.df_dict:
        for season in test_lib.df_dict[stat]:
            for region in test_lib.df_dict[stat][season]:
                df = pd.DataFrame(test_lib.df_dict[stat][season][region])
                for model in df["model"].tolist():
                    new_name = "{}-{}".format(mip.upper(), model.upper())
                    df["model"] = list(map(lambda x: x.replace(model, new_name), df["model"]))
                    if new_name not in test_models:
                        test_models.append(new_name)
                test_lib.df_dict[stat][season][region] = df
    return test_models, test_lib

def load_cmip_metrics_data(cmip_file):
    pd.set_option("future.no_silent_downcasting", True)
    cmip_lib = Metrics(cmip_file)
    cmip_models = []
    highlight_models = []
    for stat in cmip_lib.df_dict:
        for season in cmip_lib.df_dict[stat]:
            for region in cmip_lib.df_dict[stat][season]:
                df = pd.DataFrame(cmip_lib.df_dict[stat][season][region])
                for model in df["model"].tolist():
                    if model not in cmip_models:
                        cmip_models.append(model)
                    if ("e3sm" in model.lower()) and (model not in highlight_models):
                        highlight_models.append(model)
                for model in highlight_models:
                    idxs = df[df.iloc[:, 0] == model].index
                    for idx in idxs:
                        df = shift_row_to_bottom(df, idx)
                cmip_lib.df_dict[stat][season][region] = df
    return cmip_models, highlight_models, cmip_lib

def save_figure_data(stat, region, season, var_names, var_units, data_dict, template, outdir):
    fname = (
        template.replace("%(metric)", stat)
        .replace("%(region)", region)
        .replace("%(season)", season)
    )
    outfile = os.path.join(outdir, fname)
    outdic = pd.DataFrame(data_dict)
    if "model_run" in outdic.columns:
        outdic = outdic.drop(columns=["model_run"])
    for var in list(outdic.columns.values[3:]):
        if var not in var_names:
            outdic = outdic.drop(columns=[var])
        else:
            outdic.columns.values[
                outdic.columns.values.tolist().index(var)
            ] = var_units[var_names.index(var)]

    outdic.to_csv(outfile)

def construct_port4sea_axis_lables(var_names, cmip_models, test_models, highlight_models):
    model_list = cmip_models + test_models
    lable_colors = []
    for model in model_list:
        if model in highlight_models:
            lable_colors.append("#5170d7")
        elif model in test_models:
            lable_colors.append("#FC5A50")
        else:
            lable_colors.append("#000000")

    if len(model_list) > len(var_names):
        xlabels = model_list
        ylabels = var_names
        landscape = True
    else:
        xlabels = var_names
        ylabels = model_list
        landscape = False
    return xlabels, ylabels, lable_colors, landscape

def construct_port4sea_data(stat, seasons, region, data_dict, var_names, var_units, file_template, outdir, landscape):
    data_all = dict()
    for season in seasons:
        save_figure_data(
            stat, region, season, var_names, var_units, data_dict[stat][season][region], file_template, outdir
        )
        if stat == "cor_xy":
            data_nor = data_dict[stat][season][region][var_names].to_numpy()
            data_all[season] = data_nor.T if landscape else data_nor
        elif stat == "bias_xy":
            data_sea = data_dict[stat][season][region][var_names].to_numpy()
            data_rfm = data_dict["mean-obs_xy"][season][region][var_names].to_numpy()
            data_msk = np.where(np.abs(data_rfm) == 0.0, np.nan, data_rfm)
            data_nor = data_sea * 100.0 / data_msk
            data_all[season] = data_nor.T if landscape else data_nor
        else:
            data_sea = data_dict[stat][season][region][var_names].to_numpy()
            if landscape:
                data_sea = data_sea.T
                data_all[season] = normalize_by_median(data_sea, axis=1)
            else:
                data_all[season] = normalize_by_median(data_sea, axis=0)

    data_all_nor = np.stack(
        [data_all["djf"], data_all["mam"], data_all["jja"], data_all["son"]]
    )
    return data_all_nor

def port4sea_plot(stat, region, seasons, data_dict, var_names, var_units, cmip_models, test_models,
                  highlight_models, file_template, figure_template, outdir, data_version=None, watermark=False):
    fontsize = 20
    var_names = sorted(var_names)
    var_units = sorted(var_units)

    xaxis_labels, yaxis_labels, lable_colors, landscape = construct_port4sea_axis_lables(
        var_names, cmip_models, test_models, highlight_models
    )

    data_all_nor = construct_port4sea_data(
        stat, seasons, region, data_dict, var_names, var_units, file_template, outdir, landscape
    )

    if stat == "cor_xy":
        cbar_label = "Pattern Corr."
        var_range = (-1.0, 1.0)
        cmap_bounds = [0.1, 0.2, 0.4, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    elif stat == "bias_xy":
        cbar_label = "{}, relative (%)".format(stat.upper())
        var_range = (-30.0, 30.0)
        cmap_bounds = [-30.0, -20.0, -10.0, -5.0, -1, 0.0, 1.0, 5.0, 10.0, 20.0, 30.0]
    else:
        cbar_label = "{}, normalized by median".format(stat.upper())
        var_range = (-0.5, 0.5)
        cmap_bounds = [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5]

    figsize = (40, 18) if landscape else (18, 25)
    legend_box_xy = (1.08, 1.18) if landscape else (1.25, 1)
    shrink = 0.8 if landscape else 1.0

    fig, ax, cbar = portrait_plot(
        data_all_nor,
        xaxis_labels=xaxis_labels,
        yaxis_labels=yaxis_labels,
        cbar_label=cbar_label,
        cbar_label_fontsize=fontsize * 1.2,
        box_as_square=True,
        vrange=var_range,
        figsize=figsize,
        cmap="RdYlBu_r",
        cmap_bounds=cmap_bounds,
        cbar_kw={"extend": "both", "shrink": shrink},
        missing_color="white",
        legend_on=True,
        legend_labels=["DJF", "MAM", "JJA", "SON"],
        legend_box_xy=legend_box_xy,
        legend_box_size=4 if landscape else 3,
        legend_lw=1.5,
        legend_fontsize=fontsize * 0.8,
        logo_rect=[0,0,0,0],
        logo_off=True
    )

    ax.set_xticklabels(xaxis_labels, rotation=45, va="bottom", ha="left")
    ax.set_yticklabels(yaxis_labels, rotation=0, va="center", ha="right")
    if landscape:
        for xtick, color in zip(ax.get_xticklabels(), lable_colors):
            xtick.set_color(color)
    else:
        for ytick, color in zip(ax.get_yticklabels(), lable_colors):
            ytick.set_color(color)

    ax.tick_params(axis="x", labelsize=fontsize)
    ax.tick_params(axis="y", labelsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)

    ax.set_title(
        "Model Performance of Seasonal Climatology ({}, {})".format(stat.upper(), region.upper()),
        fontsize=fontsize * 1.5,
        pad=30,
    )

    figname = (
        figure_template.replace("%(metric)", stat)
        .replace("%(region)", region)
        .replace("%(season)", "4season")
    )
    plt.savefig(os.path.join(outdir, figname), facecolor="w", bbox_inches="tight")
    plt.close()

def paracord_plot(stat, region, season, data_dict, var_names, var_units, cmip_models, test_models,
                  highlight_models, file_template, figure_template, outdir, data_version=None, watermark=False):
    var_names = sorted(var_names)
    var_units = sorted(var_units)

    save_figure_data(stat, region, season, var_names, var_units, data_dict, file_template, outdir)
    model_data = data_dict[var_names].to_numpy()

    model_list = cmip_models + test_models + ["CMIP MME"]
    model_list_group2 = highlight_models + test_models
    models_to_highlight = ["CMIP MME"] + test_models

    figsize = (32, 8)
    fontsize = 20
    legend_ncol = int(11 * figsize[0] / 32.0)
    legend_posistion = (0.50, -0.14)
    colormap = "tab20_r"
    lncolors = ["#000000", "#e41a1c", "#ff7f00", "#4daf4a", "#f781bf", "#a65628", "#984ea3", "#999999", "#377eb8"]

    if not np.isnan(model_data).all():
        title = "Model Performance of {} Climatology ({}, {})".format(season.upper(), stat.upper(), region.upper())
        fig, ax = parallel_coordinate_plot(
            model_data,
            var_names,
            model_list,
            model_names2=model_list_group2,
            group1_name="CMIP MME",
            group2_name="E3SM Models",
            models_to_highlight=models_to_highlight,
            models_to_highlight_colors=lncolors,
            models_to_highlight_labels=models_to_highlight,
            title=title,
            figsize=figsize,
            colormap=colormap,
            show_boxplot=False,
            show_violin=True,
            violin_colors=("lightgrey", "pink"),
            legend_ncol=legend_ncol,
            legend_bbox_to_anchor=legend_posistion,
            legend_fontsize=fontsize * 0.6,
            xtick_labelsize=fontsize * 0.8,
            ytick_labelsize=fontsize * 0.8,
            logo_rect=[0,0,0,0],
            logo_off=True
        )

        figname = (
            figure_template.replace("%(metric)", stat)
            .replace("%(region)", region)
            .replace("%(season)", season)
        )
        plt.savefig(os.path.join(outdir, figname), facecolor="w", bbox_inches="tight")
        plt.close()

# ============================================================
# Main Diagnostic Execution Wrapper
# ============================================================
def mean_climate_metrics_plot(parameter):
    test_mip = parameter.test_data_set.split(".")[0]
    test_exp = parameter.test_data_set.split(".")[1]
    test_product = parameter.test_data_set.split(".")[2]
    test_case_id = parameter.test_data_set.split(".")[-1]

    outdir = os.path.join(parameter.output_path, test_mip, test_exp, test_case_id)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    file_template = "%(metric)_%(region)_{}_{}_{}_{}_mean_climate_%(season)_{}.csv"
    file_template = file_template.format(
        parameter.run_type.upper(),
        test_mip.upper(),
        test_exp.upper(),
        test_product.upper(),
        parameter.period,
    )
    figure_template = file_template.replace("csv", parameter.ftype)

    test_file, refr_file, cmip_file = find_metrics_data(parameter)
    cmip_models, highlight_models, cmip_lib = load_cmip_metrics_data(cmip_file)
    test_models, test_lib = load_test_model_data(test_file, refr_file, test_mip, parameter.run_type)

    test_lib, cmip_lib, var_names, var_unit_list = fill_plot_var_and_units(test_lib, cmip_lib)

    regions = []
    for reg in parameter.regions:
        if (reg in test_lib.regions) and (reg in cmip_lib.regions):
            regions.append(reg)

    merged_lib = cmip_lib.merge(test_lib)

    parall_fig_dir = os.path.join(outdir, "paracord_annual")
    if os.path.exists(parall_fig_dir):
        shutil.rmtree(parall_fig_dir)
    os.makedirs(parall_fig_dir)

    metrics_list = ["rms_xyt", "std-obs_xyt", "std_xyt", "rms_y", "rms_devzm", "std_xy_devzm", "std-obs_xy_devzm"]
    for metric in metrics_list:
        for region in regions:
            for season in ["ann"]:
                if metric in merged_lib.df_dict:
                    data_dict = merged_lib.df_dict[metric][season][region]
                    data_dict.loc["CMIP MME"] = cmip_lib.df_dict[metric][season][region].mean(numeric_only=True, skipna=True)
                    data_dict.at["CMIP MME", "model"] = "CMIP MME"
                    paracord_plot(
                        metric, region, season, data_dict, var_names, var_unit_list,
                        cmip_models, test_models, highlight_models, file_template,
                        figure_template, parall_fig_dir
                    )

    ptrait_fig_dir = os.path.join(outdir, "portrait_4seasons")
    if os.path.exists(ptrait_fig_dir):
        shutil.rmtree(ptrait_fig_dir)
    os.makedirs(ptrait_fig_dir)

    seasons = ["djf", "mam", "jja", "son"]
    data_dict = merged_lib.df_dict
    for metric in ["rms_xy", "cor_xy", "bias_xy"]:
        for region in regions:
            if metric in data_dict:
                port4sea_plot(
                    metric, region, seasons, data_dict, var_names, var_unit_list,
                    cmip_models, test_models, highlight_models, file_template,
                    figure_template, ptrait_fig_dir
                )
