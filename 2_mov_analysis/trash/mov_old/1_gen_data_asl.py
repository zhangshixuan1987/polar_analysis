#!/bin/env python
##############################################################################
# This model is used to generate mean climate diagnostic figures 
# Author: Shixuan Zhang (shixuan.zhang@pnnl.gov)
#############################################################################
import os
import glob
import sys
import json
import numpy as np
import pandas as pd 
import csv
import math
import shutil
import matplotlib as mpl
from collections import OrderedDict
from re import split
from pcmdi_metrics import stats
from pcmdi_metrics.graphics import Metrics
from pcmdi_metrics.graphics import normalize_by_median
from pcmdi_metrics.graphics import portrait_plot
from pcmdi_metrics.graphics import parallel_coordinate_plot

from mov_plot_parser import (
    create_mov_plot_parser,
    metrics_inquire,
    find_latest,
    shift_row_to_bottom,
    find_cmip_metric_data,
    select_models,
    exclude_models,
)

def load_metric_file_list(path,bases,modes,indices,period,seasons):
    file_list = {}
    for i,mode in enumerate(modes):
      refobs = bases[i]
      for sea in seasons: 
        if sea in ['monthly']: 
           ftest = glob.glob(os.path.join(path,refobs,
              "{}*{}.{}.csv".format(mode,'AC',period)))
        else: 
          ftest = glob.glob(os.path.join(path,refobs,
              "{}*{}.{}.csv".format(mode,sea,period)))
        for ff in ftest:
          if os.path.exists(ff):
            #print(ff.split("/")[-1])
            fname = ff.split("/")[-1]
            metric = fname.split(".")[1]
            if mode not in file_list.keys():
              file_list[mode] = {}
            if metric not in file_list[mode].keys():
              file_list[mode][metric] = {}
            if metric not in file_list[mode][metric].keys():
              file_list[mode][metric][sea] = {}
            file_list[mode][metric][sea] = ff 
        del (ftest)
    return file_list

def load_metrics_data_asl(csv_list,modes,indices,seasons):
    out_lib = {}
    highlight_models = []
    test_models = []
    for mode in modes:
      for metric in  csv_list[mode].keys():
        title = ['model','run','model_run']
        for ind in indices:
          for sea in csv_list[mode][metric].keys():
            metx = '{}-{}'.format(ind,sea)
            if metx not in title:
              title.append(metx)
        df = pd.DataFrame([],columns=title)
        for i,sea in enumerate(csv_list[mode][metric].keys()):
          csvObj = csv_list[mode][metric][sea]
          mod_data = pd.read_csv(csvObj,index_col=False)
          for ind in indices:
            metx = '{}-{}'.format(ind,sea)
            df[metx] = mod_data[ind]
          if i == 0:
            df['model'] = [model.replace("_r1i1p1f1","") for model in mod_data['model_run']]
            df['run'] = mod_data['run']
            df['model_run'] = mod_data['model_run']
            for k,model in enumerate(mod_data['model']):
               if ("e3sm" in model.lower()) and (model not in highlight_models):
                 highlight_models.append(model)
            for k,model in enumerate(mod_data['model_run']):
               if ("sorrm" in model.lower()) and (model not in test_models):
                 test_models.append(mod_data['model'][k])
          del(csvObj,mod_data)
        if mode not in out_lib.keys():
          out_lib[mode] = {}
        if metric not in out_lib[mode].keys():
          out_lib[mode][metric] = {}
        out_lib[mode][metric] = df ; del(df)
    return out_lib, highlight_models, test_models 

def save_figure_data(mode,stat,var_names,var_units,data_dict,template,outdir):
   # construct output file name
   fname = template.replace("%(metric)",stat).replace("%(mode)",mode.upper())
   outfile = os.path.join(outdir, fname)
   outdic = pd.DataFrame(data_dict)
   outdic = outdic.drop(columns=["model_run"])
   print(outfile)
   for var in list(outdic.columns.values[3:]):
       if var not in var_names:
           print("{} is excluded from the {}".format(var, fname))
           outdic = outdic.drop(columns=[var])
       else:
           # replace the variable with the name + units
           outdic.columns.values[
               outdic.columns.values.tolist().index(var)
           ] = var_units[var_names.index(var)]
   
   # save data to .csv file
   outdic.to_csv(outfile,index=False)
   del (fname, outfile, outdic)
   return

def main(parameter):
    modes = parameter.modes   
    indices = parameter.indices
    bases = parameter.refobs
    seasons = parameter.seasons
    # find the metrics data
    pmp_path = parameter.data_path
    period = parameter.period 

    # find the metrics data
    test_set = parameter.test_data_set
    test_mip = test_set.split(".")[0]
    test_exp = test_set.split(".")[1]
    test_product = test_set.split(".")[2]
    test_case_id = test_set.split(".")[-1]

    # output directory
    figdir = parameter.figure_path
    outdir = parameter.output_path
    if not os.path.exists(outdir):
      os.makedirs(outdir)
    
    # construct file template to save the figure data in .csv file
    file_template = "%(mode).%(metric).{}.{}.{}.{}.{}.csv"
    file_template = file_template.format(
            parameter.run_type.upper(),
            test_mip.upper(),
            test_exp.upper(),
            test_product.upper(),
            parameter.period
    )
    # construct figure template
    figure_template = file_template.replace("csv",parameter.ftype)

    # find list of metrics data files
    file_list = load_metric_file_list(pmp_path,bases,modes,indices,period,seasons)
    #load the data and reorganize if needed
    merge_dict, highlight_models, test_models = load_metrics_data_asl(file_list,modes,indices,seasons)

    #add ensemble mean 
    for mode in merge_dict.keys():
      for metric in merge_dict[mode].keys():
        data_dict = merge_dict[mode][metric]
        var_list = list(data_dict.columns[3:])
        var_unit_list = []
        for var in var_list:
          strs = var.split("-") 
          var_unit_list.append("{}-{}".format(strs[0],strs[1]))
        df_sub1 = merge_dict[mode][metric]
        df_sub2 = merge_dict[mode][metric]
        df_sub1 = df_sub1[df_sub1['run'] == "r1i1p1f1"]
        df_sub2 = df_sub2[df_sub2['run'] != "r1i1p1f1"]
        data_dict.loc['CMIP6_MME'] = df_sub1.mean(numeric_only=True, skipna=True)
        data_dict.at['CMIP6_MME','model'] = 'CMIP6'
        data_dict.at['CMIP6_MME','run'] = 'MMM'
        data_dict.at['CMIP6_MME','model_run'] = 'CMIP_MMM'
        data_dict.loc['v21_SORRM'] = df_sub2.mean(numeric_only=True, skipna=True)
        data_dict.at['v21_SORRM','model'] = 'v2_1-SORRM'
        data_dict.at['v21_SORRM','run'] = 'MMM'
        data_dict.at['v21_SORRM','model_run'] = 'v2_1-SORRM_MMM'
        data_dict = data_dict.fillna(value=np.nan)
        print("plotting {}".format(metric))
        save_figure_data(mode,metric,var_list,var_unit_list,data_dict,file_template,outdir)
        del(data_dict,df_sub1,df_sub2)

    return

if __name__ == "__main__":
  results_dir = "/lcrc/group/e3sm/ac.szhang/acme_scratch/e3sm_project/E3SMv21_testings/paper_material"
  data_dir = "/lcrc/group/e3sm/ac.szhang/acme_scratch/e3sm_project/E3SMv21_testings/paper_material"

  parser = create_mov_plot_parser()
  parameter = parser.get_parameter(argparse_vals_only=False)
  parameter.data_path = os.path.join(data_dir,"fig_data","asl_analysis","clim_index")
  parameter.period = "1950-2014"
  parameter.run_type = "model_vs_obs"
  parameter.test_product = "v2_1-SORRM"
  parameter.test_data_set = "e3sm.historical.v2_1-SORRM.v20240731"

  parameter.modes = [   'ASL'  ]
  parameter.refobs = ['NOAA-20C.00']
  parameter.seasons = ['DJF',"JJA","MAM","SON","monthly"]
  parameter.indices = ['lon','lat','ActCenPres','SectorPres','RelCenPres']

  parameter.output_path = os.path.join(results_dir,"fig_data","mov_analysis")
  parameter.figure_path = os.path.join(results_dir,"2_southern_hemisphere","mov")

  parameter.ftype = "pdf"
  parameter.debug = False

  main(parameter)

