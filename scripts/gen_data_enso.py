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
from pcmdi_metrics.graphics import Metrics
from pcmdi_metrics.graphics import normalize_by_median
from pcmdi_metrics.graphics import portrait_plot
from pcmdi_metrics.graphics import parallel_coordinate_plot
from EnsoMetrics.EnsoCollectionsLib import defCollection

from mov_plot_parser import (
    create_mov_plot_parser,
    metrics_inquire,
    find_latest,
    shift_row_to_bottom,
    find_cmip_metric_data,
    select_models,
    exclude_models,
)

def load_metric_file_list(path,mip,exp,key,case_id,modes,bases,eofs): 
    mode = modes[modes.index('ENSO')]
    base = bases[modes.index('ENSO')]
    metric_collection = eofs[modes.index('ENSO')]
    file_list = []
    if 'cmip' in mip: 
      ftest = glob.glob(os.path.join(path,mip,exp,case_id,mode,base,
          "{}_{}_{}_*_{}.json".format(mip,exp,metric_collection,key)))  
    else:
      ftest = glob.glob(os.path.join(path,mip,exp,case_id,mode,base,
          "{}_{}_{}_*_{}_*.json".format(mip,exp,metric_collection,key)))
    for ff in ftest:
      if os.path.exists(ff) and "diveDown" not in ff:
        file_list.append(ff.replace("",""))
        #print(ff.split("/")[-1])
    del (ftest)
    return file_list

def load_metrics_data_enso(mip,json_list,key,eofs,bases,modes):
    mode = modes[modes.index('ENSO')]
    base = bases[modes.index('ENSO')]
    metric_collection = eofs[modes.index('ENSO')]
    #metrics = sorted(defCollection(metric_collection)['metrics_list'].keys(), key=lambda v: v.upper())
    #print(metrics)
    #metrics = ['EnsoAmpl', 'EnsoDuration', 'EnsoSeasonality']

    #load the data and reorganize if needed
    data_lib = {}
    for jsonObj in json_list:
      with open(jsonObj) as json_file:
        dataDict = json.load(json_file)['RESULTS']['model']
        if mip == "e3sm":
          groups = [ key ]
        else:
          groups = dataDict.keys()
        for group in groups:
          if mip == "e3sm":
            exps = dataDict[group].keys()
          else:
            exps = [ key ]
          for exp in exps:
            for metric in dataDict[group][exp]["value"].keys(): 
              print(mip,group,exp,metric)  
              # get diagnostic values for the given model and observations
              if metric_collection == "ENSO_tel" and "Map" in met:
                dict_dia = dataDict[group][exp]["value"][metric+"Corr"]["diagnostic"]
                diagnostic_values = dict((key1, None) for key1 in dict_dia.keys())
                diagnostic_units = ""
              else:
                dict_dia = dataDict[group][exp]["value"][metric]["diagnostic"]
                diagnostic_values = dict((key1, dict_dia[key1]["value"]) for key1 in dict_dia.keys())
                diagnostic_units = dataDict[group][exp]["metadata"]["metrics"][metric]["diagnostic"]["units"]
              # get metric values computed with the given model and observations
              if metric_collection == "ENSO_tel" and "Map" in met:
                list1, list2 = [metric+"Corr", metric+"Rmse"], ["diagnostic", "metric"]
                dict_met = dataDict[group][exp]["value"]
                metric_values = dict((key1, {model: [dict_met[su][ty][key1]["value"] for su, ty in zip(list1, list2)]})
                                 for key1 in dict_met[list1[0]]["metric"].keys())
                metric_units = [dataDict[group][exp]["metadata"]["metrics"][su]["metric"]["units"] for su in list1]
              else:
                dict_met = dataDict[group][exp]["value"][metric]["metric"]
                metric_values = dict((key1, {exp: dict_met[key1]["value"]}) for key1 in dict_met.keys())
                metric_units = dataDict[group][exp]["metadata"]["metrics"][metric]["metric"]["units"]
              print(diagnostic_values,diagnostic_units)
              print(metric_values,metric_units)
              if metric not in data_lib.keys():
                data_lib[metric] = {}
              if group not in data_lib[metric].keys():
                data_lib[metric][group] ={}
              if exp not in data_lib[metric][group].keys():
                data_lib[metric][group][exp] = {}
              for ff in diagnostic_values.keys():
                gg = "{}_{}".format(group,exp)
                print(metric,group,exp,ff,gg)
                if ff == gg:
                  data_lib[metric][group][exp]['diagnostic_model'] = diagnostic_values[gg]
                  data_lib[metric][group][exp]['diagnostic_unit'] = diagnostic_units
                else:
                  data_lib[metric][group][exp]['diagnostic_obs'] = diagnostic_values[ff]  
                  data_lib[metric][group][exp]['metric_value'] = metric_values[ff][exp] 
                  data_lib[metric][group][exp]['metric_unit'] = metric_units
              del(metric_values,metric_units,diagnostic_values,diagnostic_units)

    out_lib = {}
    highlight_models = []
    for metric in data_lib.keys():
      if metric not in out_lib.keys():
        out_lib[metric] = {}
      title = ['model','run','model_run']
      for group in data_lib[metric].keys():
        for exp in data_lib[metric][group].keys():
          dtmp = []
          dtmp.append(group)
          dtmp.append(exp)
          dtmp.append('{}_{}'.format(group,exp))
          title = ['model','run','model_run']
          for mode in ['diagnostic_obs','diagnostic_model','metric_value']:
            dtmp.append(data_lib[metric][group][exp][mode])
            if 'diagnostic' in mode:
              key  = '{}({})'.format(mode,data_lib[metric][group][exp]['diagnostic_unit'])
            else:
              key  = '{}({})'.format(mode,data_lib[metric][group][exp]['metric_unit'])
            if key not in title: 
              title.append(key)
          if 'df' not in locals():
            df = pd.DataFrame([dtmp],columns=title)
          else:
            df = pd.concat([df, pd.DataFrame([dtmp],columns=title)],ignore_index=True)
          del(dtmp) 

      if mip == "cmip":
        for model in df['model']:
          if ("e3sm" in model.lower()) and (model not in highlight_models):
                  highlight_models.append(model)

        for model in highlight_models:
          idxs = df[df.iloc[:,0] == model].index
          for idx in idxs:
            igx = [i for i in df.index if i != idx]
            df =  df.loc[igx + [idx]] ; del(igx)
          del(idxs)

      out_lib[metric] = df ; del(df,title)
    return out_lib, highlight_models

def save_figure_data(stat,var_names,var_units,data_dict,template,outdir):
   # construct output file name
   fname = template.replace("%(metric)",stat)
   outfile = os.path.join(outdir, fname)
   outdic = pd.DataFrame(data_dict)
   outdic = outdic.drop(columns=["model_run"])
   if os.path.exists(outfile):
     os.remove(outfile)  
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
    eofs  = parameter.eofs
    bases = parameter.bases

    # find the metrics data
    pmp_path = parameter.pcmdi_data_path
    cmip_set = parameter.pcmdi_data_set
    test_set = parameter.test_data_set
    refr_set = parameter.refr_data_set
    run_type = parameter.run_type
    debug = parameter.debug

    cmip_mip = cmip_set.split(".")[0]
    cmip_exp = cmip_set.split(".")[1]
    cmip_product = cmip_set.split(".")[2]
    cmip_case_id = cmip_set.split(".")[-1]

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
    file_template = "ENSO.%(metric).{}.{}.{}.{}.{}.csv"
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
    test_list = load_metric_file_list(pmp_path,test_mip,test_exp,test_product,test_case_id,modes,bases,eofs)
    #load the data and reorganize if needed
    test_lib, dummy = load_metrics_data_enso('e3sm',test_list,test_product,eofs,bases,modes)
    #print(test_lib)

    # find list of metrics data files
    cmip_list = load_metric_file_list(pmp_path,cmip_mip,cmip_exp,'r1i1p1f1',cmip_case_id,modes,bases,eofs)
    #load the data and reorganize if needed
    cmip_lib, highlight_models = load_metrics_data_enso('cmip',cmip_list,cmip_product,eofs,bases,modes)

    #model_vs_model, merge the reference model data into test model
    if run_type == "model_vs_model":
      refr_mip = test_set.split(".")[0]
      refr_exp = test_set.split(".")[1]
      refr_product = test_set.split(".")[2]
      refr_case_id = test_set.split(".")[-1]
      refr_list = load_metric_file_list(pmp_path,refr_mip,refr_exp,refr_group,refr_case_id,modes,bases,eofs)
      refr_lib = load_metrics_data_enso('e3sm',refr_list,refr_product,eofs,bases,modes)
      for metric in test_lib.keys():
        test_lib[metric] = pd.concat([test_lib[metric],refr_lib[metric]],ignore_index=True)
      del(refr_lib,refr_list,refr_case_id,refr_product,refr_exp,refr_mip)

    #merge cmip and test:
    merge_dict = {}
    cmip_models = []
    test_models = []
    for metric in test_lib:
      if metric not in merge_dict.keys():
        merge_dict[metric] = {}
      df1 = cmip_lib[metric]
      df2 = test_lib[metric]
      for model in df1['model']:
        if model not in cmip_models:
          cmip_models.append(model)
      for i,model in enumerate(df2["model_run"].tolist()):
        if "v2_1-SORRM" in model:
          new_name = model.replace("v2_1-SORRM","v2_1-SORRM")
          idxs = df2[df2.iloc[:,2] == model ].index
          df2.loc[idxs,"model"] = list(
                map(lambda x: x.replace("v2_1-SORRM", new_name),df2.loc[idxs,"model"],)
                )
          if new_name not in test_models:
            test_models.append(new_name)
        else:
          test_models.append(model)
      merge_dict[metric] = pd.concat([df1,df2],ignore_index=True)
      del(df1,df2)
      
    #add ensemble mean 
    for metric in merge_dict.keys():
      data_dict = merge_dict[metric]
      df_sub2 = merge_dict[metric]
      data_dict = merge_dict[metric]
      var_list = list(data_dict.columns[3:])
      var_unit_list = []
      for var in var_list:
        strs = var.replace("("," ").replace(")","").split(" ")
        var_unit_list.append("{}-{}".format(strs[0],strs[1]))
      for i,model in enumerate(test_models):
        df_sub2 = df_sub2[df_sub2["model"] != model]
        if i == 0:
          df_sub1 = data_dict.loc[data_dict["model"] == model]
        else:
          tmp_sub1 = data_dict.loc[data_dict["model"] == model]
          df_sub1 = pd.concat([df_sub1,tmp_sub1])
          del(tmp_sub1)
      data_dict.loc['CMIP6_MME'] = df_sub2.mean(numeric_only=True, skipna=True)
      data_dict.at['CMIP6_MME','model'] = 'CMIP6'
      data_dict.at['CMIP6_MME','run'] = 'MMM'
      data_dict.at['CMIP6_MME','model_run'] = 'CMIP6_MMM'
      data_dict.loc['v21_SORRM'] = df_sub1.mean(numeric_only=True, skipna=True)
      data_dict.at['v21_SORRM','model'] = 'v2_1-SORRM'
      data_dict.at['v21_SORRM','run'] = 'MMM'
      data_dict.at['v21_SORRM','model_run'] = 'v2_1-SORRM_MMM'
      data_dict = data_dict.fillna(value=np.nan)
      print("plotting {}".format(metric))
      print(data_dict)
      save_figure_data(metric,var_list,var_unit_list,data_dict,file_template,outdir)
      del(data_dict,df_sub1,df_sub2)

    return

if __name__ == "__main__":
  
  results_dir = "/lcrc/group/e3sm/public_html/diagnostic_output/ac.szhang/polar_analysis/data"
  data_dir = "/lcrc/group/e3sm/public_html/diagnostic_output/ac.szhang/E3SMv2_1/pcmdi"

  parser = create_mov_plot_parser()
  parameter = parser.get_parameter(argparse_vals_only=False)
  parameter.pcmdi_data_path = os.path.join(data_dir,"metrics_results","variability_modes")
  parameter.pcmdi_data_set = "cmip6.historical.r1i1p1f1.v20241111"
  parameter.period = "1950-2014"
  parameter.test_product = "v2_1-SORRM"
  parameter.test_data_set = "e3sm.historical.v2_1-SORRM.v20241111"

  parameter.run_type = "model_vs_obs"
  parameter.refr_data_set = ""
  parameter.refr_data_path = ""
  #parameter.modes = ['AMO','SAM','PSA1','PSA2','ASL']
  #parameter.eofs  = ['EOF1','EOF1','EOF2','EOF3','ASL']
  #parameter.bases = ['HadSST2','NOAA-20C','NOAA-20C','NOAA-20C','NOAA-20C']
  parameter.modes = ['ENSO',      'AMO',    'SAM',     'PSA1',    'PSA2',    'ASL'     ]
  parameter.eofs  = ['ENSO_perf', 'EOF1',   'EOF1',    'EOF2',    'EOF3',    'ASL'     ]
  parameter.bases = ['default','HadSST2','NOAA-20C','NOAA-20C','NOAA-20C','NOAA-20C']
  if parameter.run_type == "model_vs_model":
    parameter.refr_data_set = "e3sm.historical.v2_1-LR.v20241111"
    parameter.refr_data_path = os.path.join(results_dir,"metrics_results","variability_modes")

  parameter.output_path = os.path.join(results_dir,"fig_data","mov_analysis")
  parameter.figure_path = os.path.join(results_dir,"2_southern_hemisphere","mov")
  parameter.ftype = "pdf"
  parameter.debug = False

  main(parameter)

