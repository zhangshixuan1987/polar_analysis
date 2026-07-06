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

def load_metrics_data_asl(path,template,mode,diags,metrics,units,seasons):
    title = ['model','run']
    for i,diag in enumerate(diags):
      for j,metric in enumerate(metrics):
        for sea in seasons:
          if metric == "bias":
            metx = "{}({})-{}({})".format(mode,diag,"bias",units[i])
          elif metric == "rms_xyt":
            metx = "{}({})-{}({})".format(mode,diag,"rmse",units[i])
          else:
            metx = diag
          if metx not in title:
            title.append(metx)
    out_lib = pd.DataFrame([],columns=title)
    for i,metric in enumerate(metrics):
      fin = os.path.join(path,template.replace("%(mode)",mode).replace("%(metric)",metric))
      fdat = pd.read_csv(fin,index_col=False)
      if i == 0:
        out_lib['model'] = fdat['model']
        out_lib['run'] = fdat['run']
      for sea in seasons:
        for j,diag in enumerate(diags):
          if metric == "bias":
            metx = "{}({})-{}({})".format(mode,diag,"bias",units[j])
          elif metric == "rms_xyt":
            metx = "{}({})-{}({})".format(mode,diag,"rmse",units[j])
          else:
            metx = diag
          if sea in ['monthly', "Monthly", "MON", "MONTHLY"]:
            key = "{}-{}".format(diag,'monthly')
          else:
            key = "{}-{}".format(diag,sea)
          print(metx,key)
          if key in fdat.columns:
            out_lib[metx] = fdat[key]
            if metric == "frac":
              out_lib[metx] = out_lib[metx] * 100.0
          else:
            exit("metric data {} not exist, please check....".format(key))
    #remove bad model data         
    for i,model in enumerate(out_lib["model"].tolist()):
      if model in ['BCC-CSM2-MR','CAS-ESM2-0']:
        idxs = out_lib[out_lib.iloc[:,0] == model].index
        for metx in out_lib.columns:
           if 'ASL(lon)' in metx:
              out_lib.loc[idxs,metx] = np.nan
      if model in [ 'GISS-E2-1-G']: 
        idxs = out_lib[out_lib.iloc[:,0] == model].index
        for metx in out_lib.columns:
           if 'ASL(lat)' in metx:
              out_lib.loc[idxs,metx] = np.nan   

        del(idxs)            
    return out_lib

def load_metrics_data_mov(path,template,mode,diags,metrics,units,seasons):
    title = ['model','run']
    for j,metric in enumerate(metrics):
      for i,diag in enumerate(diags):
        for sea in seasons:
          k = j*len(diags) + i 
          if metric == "frac":
            metx = "{}-{}({})".format(diag,"evar",units[k])
          elif metric == "rms":
            metx = "{}-{}({})".format(diag,"rmse",units[k])
          elif metric == "cor":
            metx = "{}-{}({})".format(diag,"pcor","1")
          else:
            metx = diag
          if metx not in title:
            title.append(metx)
    out_lib = pd.DataFrame([],columns=title)
    for i,metric in enumerate(metrics):
      fin = os.path.join(path,template.replace("%(mode)",mode).replace("%(metric)",metric))
      fdat = pd.read_csv(fin,index_col=False)
      if i == 0:
        out_lib['model'] = fdat['model']
        out_lib['run'] = fdat['run']
      for diag in diags:
        for sea in seasons:
          k = i*len(metrics) + j
          if metric == "frac":
            metx = "{}-{}({})".format(diag,"evar",units[k])
          elif metric == "rms":
            metx = "{}-{}({})".format(diag,"rmse",units[k])
          elif metric == "cor":
            metx = "{}-{}({})".format(diag,"pcor","1")
          else:
            metx = diag
          if sea in ['monthly', "Monthly", "MON", "MONTHLY"]:
            key = "{}-{}".format(diag,'monthly') 
          else:
            key = "{}-{}".format(diag,sea)  
          print(metx,key)
          if key in fdat.columns:
            out_lib[metx] = fdat[key]
            if metric == "frac":
              out_lib[metx] = out_lib[metx] * 100.0  
          else:
            exit("metric data {} not exist, please check....".format(key))
    #remove bad model data         
    for i,model in enumerate(out_lib["model"].tolist()):
      if model in ['IPSL-CM5A2-INCA']:
        idxs = out_lib[out_lib.iloc[:,0] == model].index
        for metx in out_lib.columns:
           if 'SAM' in metx:
              out_lib.loc[idxs,metx] = np.nan
        del(idxs)
    return out_lib

def load_metrics_data_enso(path,template,mode,diags,metrics,units):
    title = ['model','run']
    for i,diag in enumerate(diags):
      for metric in metrics: 
        if metric == "metric_value":  
          metx = "{}-{}({})".format(diag,"bias",units[i])
        elif metric == "diagnostic_model":
          metx = "{}-{}({})".format(diag,"model",units[i])      
        elif metric == "diagnostic_obs":
          metx = "{}-{}({})".format(diag,"obs",units[i])      
        else:
          metx = diag
        if metx not in title:
          title.append(metx)
    out_lib = pd.DataFrame([],columns=title)
    for i,diag in enumerate(diags):
      fin = os.path.join(path,template.replace("%(mode)",mode).replace("%(metric)",diag))  
      fdat = pd.read_csv(fin,index_col=False)
      if i == 0: 
        out_lib['model'] = fdat['model']
        out_lib['run'] = fdat['run']
      for metric in metrics:
        if metric == "metric_value":
          metx = "{}-{}".format(diag,"bias(%)")
        elif metric == "diagnostic_model":
          metx = "{}-{}".format(diag,"model(degC)")
        elif metric == "diagnostic_obs":
          metx = "{}-{}".format(diag,"obs(degC)")
        else:
          metx = diag
        for key in fdat.columns[3:]:
          if metric in key:
            out_lib[metx] = fdat[key]
    #remove bad model data         
    for i,model in enumerate(out_lib["model"].tolist()):
      if model in ['MCM-UA-1-0','CMCC-CM2-SR5','MPI-ESM1-2-HR']:
        idxs = out_lib[out_lib.iloc[:,0] == model].index
        for metx in out_lib.columns: 
           if 'EnsoDuration' in metx: 
              out_lib.loc[idxs,metx] = np.nan
        del(idxs)
    return out_lib

def save_figure_data(stat,var_names,var_units,data_dict,template,outdir):
   # construct output file name
   fname = template.replace("%(metric)",stat)
   outfile = os.path.join(outdir, fname)
   outdic = pd.DataFrame(data_dict)
   outdic = outdic.drop(columns=["model_run"])
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
   outdic.to_csv(outfile)
   del (fname, outfile, outdic)
   return

def paracord_plot(data_dict,var_names,var_units,
                  model_list,cmip_models,test_models,highlight_models,
                  file_template,figure_template,outdir,
                  data_version=None, watermark=False):
    
    #construct plotting data
    #var_names = sorted(var_names)
    #var_units = sorted(var_units)
    #print(var_names)
    #exit()
    # write out the results as a table
    fname = file_template.replace("%(mode)",'MOV').replace("%(metric)","variability")
    outfile = os.path.join(outdir, fname)
    # save data to .csv file
    data_dict.to_csv(outfile)
    
    print(data_dict[var_names])
    model_data = data_dict[var_names].to_numpy()
    #construct the string for plot
    model_list_group2 = list(highlight_models) + list(test_models)
    models_to_highlight = ['CMIP6(MMM)','v2_1-SORRM(Mean)'] 
    print(model_list,model_list_group2)
    print(model_data)

    figsize = (45, 12)
    fontsize = 24
    legend_ncol = int(7*figsize[0]/40.0)
    legend_posistion = (0.50, -0.14)

    #color map for markers 
    colormap = "tab20_r"
    #color map for highlight lines 
    lncolors =  ['#000000', '#e41a1c', '#ff7f00', '#4daf4a','#f781bf', 
                 '#a65628', '#984ea3','#999999', '#377eb8', '#dede00']

    # Add Watermark/Logo
    if watermark:
      logo_rect = [0.85, 0.15, 0.07, 0.07]
      logo_off  = False
    else:
      logo_rect = [0, 0, 0, 0]
      logo_off  = True

    xlabel = "Metric"
    ylabel = "Metrics Value"
    title = "Spatio-temporal Error of Mode Varibility (Southern Hemisphere)"
    fig, ax = parallel_coordinate_plot(model_data,var_units,model_list,
                                       model_names2=model_list_group2,
                                       group1_name='CMIP MME',
                                       group2_name='E3SM Models',
                                       models_to_highlight = models_to_highlight,
                                       models_to_highlight_colors = lncolors,
                                       models_to_highlight_labels = models_to_highlight,
                                       #ymax=5,
                                       #ymin=-5,
                                       title=title,
                                       figsize=figsize,
                                       axes_labelsize = fontsize*1.1,
                                       title_fontsize = fontsize*1.1,
                                       yaxes_label = ylabel,
                                       xaxes_label = xlabel,
                                       colormap=colormap,
                                       show_boxplot=False,
                                       show_violin=True,
                                       violin_colors=('lightgrey', 'pink'),
                                       legend_ncol = legend_ncol,
                                       legend_bbox_to_anchor = legend_posistion,
                                       legend_fontsize = fontsize*0.85,
                                       xtick_labelsize = fontsize*0.95,
                                       ytick_labelsize = fontsize*0.95,
                                       logo_rect = logo_rect,
                                       logo_off  = logo_off)

    # Add Watermark
    if watermark:
      ax.text(0.5, 0.5, 'E3SM-PCMDI', transform=ax.transAxes,
              fontsize=100, color='black', alpha=0.5,
              ha='center', va='center', rotation=25)
      # Add data info
      fig.text(1.25, 0.9, 'Data version\n'+data_version, transform=ax.transAxes,
               fontsize=12, color='black', alpha=0.6, ha='left', va='top',)

    # Save figure as an image file
    figname = figure_template.replace("%(mode)",'MOV').replace("%(metric)","variability")
    figfile = os.path.join(outdir,figname)
    fig.savefig(figfile, facecolor='w', bbox_inches='tight')

    del(model_data,model_list,model_list_group2,models_to_highlight)

    return 

def main(parameter):
    modes = parameter.modes
    eofs  = parameter.eofs
    bases = parameter.bases
    pmp_path = parameter.data_path

    # find the metrics data
    test_set = parameter.test_data_set
    run_type = parameter.run_type
    debug = parameter.debug

    test_mip = test_set.split(".")[0]
    test_exp = test_set.split(".")[1]
    test_product = test_set.split(".")[2]
    test_case_id = test_set.split(".")[-1]

    # output directory
    outdir = parameter.output_path #os.path.join(parameter.output_path,test_mip,test_exp,test_case_id)
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
 
    #load the data and reorganize if needed
    asl_diags = parameter.asl_diags.split(',') #replace(","," ").split()
    asl_metrics = parameter.asl_metric.split(',') #replace(","," ").split()
    asl_units = parameter.asl_units.split(',')
    asl_seasons = parameter.asl_seasons.split(',')
    asl_lib  = load_metrics_data_asl(pmp_path,file_template,'ASL',asl_diags,asl_metrics,asl_units,asl_seasons)

    mov_diags = parameter.mov_diags.split(',') #replace(","," ").split()
    mov_metrics = parameter.mov_metric.split(',') #replace(","," ").split()
    mov_units = parameter.mov_units.split(',')
    mov_seasons = parameter.mov_seasons.split(',')
    mov_lib  = load_metrics_data_mov(pmp_path,file_template,'MOV',mov_diags,mov_metrics,mov_units,mov_seasons)

    enso_diags = parameter.enso_diags.split(',') #replace(","," ").split()
    enso_metrics = parameter.enso_metric.split(',') #replace(","," ").split()
    enso_units = parameter.enso_units.split(',')
    enso_lib = load_metrics_data_enso(pmp_path,file_template,'ENSO',enso_diags,enso_metrics,enso_units)
    
    #var_list  = ['model','run'] + list(enso_lib.columns[2:]) + list(mov_lib.columns[2:]) + list(asl_lib.columns[2:])
    #var_names = var_list #[var.split("-")[0] for var in var_list]
    #var_units = [var.replace("-"," \n ") for var in var_list]
    #merge_tmp = pd.merge(enso_lib,mov_lib,how='inner',on=['model','run'])
    #merge_lib = pd.merge(merge_tmp,asl_lib,how='inner',on=['model','run'])
    var_list  = ['model','run'] + list(enso_lib.columns[2:]) + list(asl_lib.columns[2:]) + list(mov_lib.columns[2:])
    var_names = var_list #[var.split("-")[0] for var in var_list]
    var_units = [var.replace("-"," \n ") for var in var_list]
    merge_tmp = pd.merge(enso_lib,asl_lib,how='inner',on=['model','run'])
    merge_lib = pd.merge(merge_tmp,mov_lib,how='inner',on=['model','run'])

    merge_lib = merge_lib.set_axis(var_names, axis='columns')
    data_dict = merge_lib[merge_lib['run'] != "MMM"].copy()
    cmip_models = data_dict[data_dict['run'] == "r1i1p1f1"]['model']
    test_models = data_dict[data_dict['run'] != "r1i1p1f1"]['model']
    highlight_models = [model for model in cmip_models if "E3SM" in model] 
    data_dict.loc['CMIP6_MME'] = data_dict[data_dict['run'] == "r1i1p1f1"].mean(numeric_only=True, skipna=True)
    data_dict.at['CMIP6_MME','model'] = 'CMIP6(MMM)'
    data_dict.at['CMIP6_MME','run'] = 'MMM'
    data_dict.loc['v21_SORRM'] = data_dict[data_dict['run'] != "r1i1p1f1"].mean(numeric_only=True, skipna=True)
    data_dict.at['v21_SORRM','model'] = 'v2_1-SORRM(Mean)'
    data_dict.at['v21_SORRM','run'] = 'MMM'
    del(merge_tmp,merge_lib,enso_lib,mov_lib,asl_lib)
    
    ###################################
    #generate parallel coordinate plot
    ###################################
    parall_fig_dir = os.path.join(outdir,"paracord_annual")
    if os.path.exists(parall_fig_dir):
      shutil.rmtree(parall_fig_dir)
    os.makedirs(parall_fig_dir)
    var_names.remove('model')
    var_names.remove('run')
    var_units.remove('model')
    var_units.remove('run')
    model_list = list(data_dict['model'])
    data_dict = data_dict.drop(columns=['model','run'])
    data_dict = data_dict.fillna(value=np.nan)
    data_dict = data_dict.dropna(thresh=2)
    paracord_plot(data_dict,var_names,var_units,
                  model_list,cmip_models,test_models,highlight_models,
                  file_template,figure_template,parall_fig_dir,
                  data_version=None,watermark=False)
    del(data_dict)

    return

if __name__ == "__main__":
  
  results_dir = "/lcrc/group/e3sm/ac.szhang/acme_scratch/e3sm_project/E3SMv21_testings/paper_material"
  data_dir = os.path.join(results_dir,"fig_data","mov_analysis")
  fig_dir = os.path.join(results_dir,"2_mov_analysis","figure")
  
  parser = create_mov_plot_parser()
  parameter = parser.get_parameter(argparse_vals_only=False)
  parameter.data_path = data_dir
  parameter.period = "1950-2014"
  parameter.test_product = "v2_1-SORRM"
  parameter.test_data_set = "e3sm.historical.v2_1-SORRM.v20240731"
  parameter.run_type = "model_vs_obs"
  
  parameter.enso_diags = 'EnsoDuration,EnsoAmpl,EnsoSeasonality'
  parameter.enso_base = 'default,default,default'
  parameter.enso_metric = 'metric_value'
  parameter.enso_units = '%,%,%'

  parameter.mov_diags = 'SAM,PSA1,PSA2'
  parameter.mov_base = 'HadSST2,NOAA-20C,NOAA-20C,NOAA-20C'
  parameter.mov_seasons = 'Monthly' 
  parameter.mov_metric = 'cor,frac,rms'
  parameter.mov_units = '1,1,1,%,%,%,hPa,hPa,hPa'

  parameter.asl_diags = "lon,lat,ActCenPres"
  parameter.asl_base = 'NOAA-20C'
  parameter.asl_seasons = 'Monthly' 
  parameter.asl_metric = 'rms_xyt'
  parameter.asl_units = 'deg,deg,hPa'

  parameter.output_path = fig_dir
  parameter.ftype = "pdf"
  parameter.debug = False

  main(parameter)

