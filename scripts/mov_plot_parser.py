#!/usr/bin/env python
import ast
import glob
import os
import pandas as pd
import numpy as np 

from pcmdi_metrics.mean_climate.lib import pmp_parser

def create_mov_plot_parser():
    parser = pmp_parser.PMPMetricsParser()
    parser.add_argument(
        "--test_model",
        dest="test_model",
        help="Defines target model for the metrics plots",
        required=False,
    )

    parser.add_argument(
        "--test_data_set",
        type=str,
        nargs="+",
        dest="test_data_set",
        help="List of observations or models to test "
        + "against the reference_data_set",
        required=False,
    )

    parser.add_argument(
        "--test_data_path",
        dest="test_data_path",
        help="Path for the test climitologies",
        required=False,
    )

    parser.add_argument(
        "--period", dest="period", help="A simulation parameter", required=False
    )

    parser.add_argument(
        "--run_type", dest="run_type", help="A post-process parameter", required=False
    )

    parser.add_argument(
        "--modes",
        type=ast.literal_eval,
        dest="modes",
        help="large-scale mode names on which to run the metrics",
        required=False,
    )

    parser.add_argument(
        "--eofs",
        type=ast.literal_eval,
        dest="eofs",
        help="numer of eof modes for corresponding mode varibility",
        required=False,
    )

    parser.add_argument(
        "--bases",
        type=ast.literal_eval,
        dest="bases",
        help="source to derive the reference(base) mode varibility",
        required=False,
    )

    parser.add_argument(
        "--pcmdi_data_set",
        type=str,
        nargs="+",
        dest="pcmdi_data_set",
        help="PCMDI CMIP dataset that is used as a "
        + "CMIP multi-model ensembles against the test_data_set",
        required=False,
    )

    parser.add_argument(
        "--pcmdi_data_path",
        dest="pcmdi_data_path",
        help="Path for the PCMDI CMIP mean climate metrics data",
        required=False,
    )

    parser.add_argument(
        "--refr_model",
        dest="refr_model",
        help="A simulation parameter",
        required=False,
    )

    parser.add_argument(
        "--refr_data_set",
        type=str,
        nargs="+",
        dest="refr_data_set",
        help="List of reference models to test " + "against the reference_data_set",
        required=False,
    )

    parser.add_argument(
        "--refr_data_path",
        dest="refr_data_path",
        help="Path for the reference model climitologies",
        required=False,
    )

    parser.add_argument(
        "--output_path",
        dest="output_path",
        help="Path for the metrics plots",
        required=False,
    )

    return parser


def metrics_inquire(name):
    # list of metrics name and long-name
    metrics = {
        "eof": "linear regressed eof pattern (eof domain) [2d field]",
        "eof_lr": "linear regressed eof pattern (global) [2d field]",
        "slope": "slope from above linear regression (bring it here to calculate rmsc) [2d field]",
        "pc": "principle component time series [1d field]",
        "stdv_pc": "standard deviation of principle component time series [float]",
        "frac" : "fraction of explained variance [1 number field]",
        "eof_obs": "eof pattern over subdomain from observation [2d field]",
        "eof_lr_obs": "eof pattern over globe (linear regressed) from observation [2d field]",
        "stdv_pc_obs": "standard deviation of principle component time series of observation [float]",
        "rms": "Root Mean Square Error of EOF pattern",
        "rms_glo": "Root Mean Square Error of Linear Regression Pattern (global)",
        "rmsc": "Root Mean Square Error of EOF pattern (centered)",
        "rmsc_glo": "Root Mean Square Error of Linear Regression Pattern (global,centered)",
        "bias": "Mean Bias (Model - Reference) of EOF pattern",
        "bias_glo": "Mean Bias (Model - Reference) of linear regression pattern (global)",
        "cor": "Spatial Pattern Correlation Coefficient of EOF pattern",
        "cor_glo": "Spatial Pattern Correlation Coefficient of linear regression pattern (global)",
        "stdv_pc_ratio_to_obs": "stdv_pc / stdv_pc_obs", 
        "tcor_cbf_vs_eof_pc": "Temporal correlation between CBF PC timeseries and usual model PC timeseries",

    }
    if name in metrics.keys():
        long_name = metrics[name]

    return long_name


def find_latest(pmprdir, mip, exp):
    versions = sorted(
        [
            r.split("/")[-1]
            for r in glob.glob(os.path.join(pmprdir, mip, exp, "v????????"))
        ]
    )
    latest_version = versions[-1]
    return latest_version


def shift_row_to_bottom(df, index_to_shift):
    idx = [i for i in df.index if i != index_to_shift]
    return df.loc[idx + [index_to_shift]]


def find_cmip_metric_data(pmprdir, data_set, var):
    # cmip data for comparison
    mip = data_set.split(".")[0]
    exp = data_set.split(".")[1]
    case_id = data_set.split(".")[2]
    if case_id == "":
        case_id = find_latest(pmprdir, mip, exp)
    fpath = glob.glob(os.path.join(pmprdir, mip, exp, case_id, "{}.*.json".format(var)))
    if len(fpath) < 1 and var == "rtmt":
        fpath = glob.glob(
            os.path.join(pmprdir, mip, exp, case_id, "{}.*.json".format("rt"))
        )
    if len(fpath) > 0 and os.path.exists(fpath[0]):
        cmip_list = fpath[0]
        return_code = 0
    else:
        print("Warning: cmip metrics data not found for {}....".format(var))
        print("Warning: remove {} from the metric list....".format(var))
        cmip_list = None
        return_code = -99
    return cmip_list, return_code


def select_models(df, selected_models):
    # Selected models only
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
    # eclude models
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

