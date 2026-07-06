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
            metx = "ASL-{}({})-{}({})".format(diag,sea,"bias",units[i])
          elif metric == "rms_xyt":
            metx = "ASL-{}({})-{}({})".format(diag,sea,"rmse",units[i])
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
            metx = "ASL-{}({})-{}({})".format(diag,sea,"bias",units[j])
          elif metric == "rms_xyt":
            metx = "ASL-{}({})-{}({})".format(diag,sea,"rmse",units[j])
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
    return out_lib

def load_metrics_data_mov(path,template,mode,diags,metrics,units,seasons):
    title = ['model','run']
    for j,metric in enumerate(metrics):
      for i,diag in enumerate(diags):
        for sea in seasons:
          k = j*len(diags) + i 
          if metric == "frac":
            metx = "{}({})-{}({})".format(diag,sea,"evar",units[k])
          elif metric == "rms":
            metx = "{}({})-{}({})".format(diag,sea,"rmse",units[k])
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
            metx = "{}({})-{}({})".format(diag,sea,"evar",units[k])
          elif metric == "rms":
            metx = "{}({})-{}({})".format(diag,sea,"rmse",units[k])
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

def construct_port4sea_axis_lables(
    var_names, cmip_models, test_models, highlight_models
):
    model_list = cmip_models + test_models
    # assign colors for labels of models
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
    del model_list
    return xlabels, ylabels, lable_colors, landscape


def construct_port4sea_data(
    stat,
    seasons,
    data_dict,
    var_names,
    var_units,
    file_template,
    outdir,
    landscape,
):
    # work array
    data_all = dict()
    # loop 4 seasons and collect data
    for season in seasons:
        # save raw metric results as a .csv file for each season
        save_figure_data(
            stat,
            var_names,
            var_units,
            data_dict,
            file_template,
            outdir,
        )
        data_sea = data_dict[stat][season][region][var_names].to_numpy()
        if landscape:
            data_sea = data_sea.T
            data_all[season] = normalize_by_median(data_sea, axis=1)
        else:
            data_all[season] = normalize_by_median(data_sea, axis=0)
        del data_sea

    # data for final plot
    data_all_nor = np.stack(
        [data_all["djf"], data_all["mam"], data_all["jja"], data_all["son"]]
    ); del data_all
    return data_all_nor

def port4sea_plot(stat,seasons,data_dict,var_names,var_units,
                  cmip_models,test_models,highlight_models,
                  file_template,figure_template,outdir,
                  data_version=None,watermark=False):

    #process figure
    fontsize = 20
    # construct the axis labels and colors
    (
        xaxis_labels,
        yaxis_labels,
        lable_colors,
        landscape,
    ) = construct_port4sea_axis_lables(
        var_names, cmip_models, test_models, highlight_models
    )

    # construct data for plotting
    data_all_nor = construct_port4sea_data(
        stat,
        seasons,
        data_dict,
        var_names,
        var_units,
        file_template,
        outdir,
        landscape,
    )

    if stat == "cor_xy":
      cbar_label   = "Pattern Corr."
      var_range    = (-1,1)
      cmap_bounds  = [0.1, 0.2, 0.4, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,1.0]
    elif stat == "bias_xy":
      cbar_label   = "{}, relative (%)".format(stat.upper())
      var_range    = (-30.,30.)
      cmap_bounds  = [-30., -20., -10., -5., -1, .0, 1., 5., 10., 20., 30.]
    else:
      cbar_label   = "{}, normalized by median".format(stat.upper())
      var_range    = (-0.5,0.5)
      cmap_bounds  = [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5]

    if landscape:
        figsize = (40, 18)
        legend_box_xy = (1.02, 1.18)
        legend_box_size = 4
        legend_lw = 1.5
        shrink = 0.8
        legend_fontsize = fontsize * 0.8
    else:
        figsize = (18, 25)
        legend_box_xy = (1.25, 1)
        legend_box_size = 3
        legend_lw = 1.5
        shrink = 1.0
        legend_fontsize = fontsize * 0.8

    # Add Watermark/Logo
    if watermark:
      logo_rect = [0.85, 0.15, 0.07, 0.07]
      logo_off  = False
    else:
      logo_rect = [0, 0, 0, 0]
      logo_off  = True

    #Using Matplotlib-based PMP Visualization Function to Generate Portrait Plot
    fig, ax, cbar = portrait_plot(data_all_nor,
                                  xaxis_labels=xaxis_labels,
                                  yaxis_labels=yaxis_labels,
                                  cbar_label=cbar_label,
                                  cbar_label_fontsize=fontsize*1.2,
                                  box_as_square=True,
                                  vrange=var_range,
                                  figsize=figsize,
                                  cmap='RdYlBu_r',
                                  cmap_bounds=cmap_bounds,
                                  cbar_kw={"extend": "both",
                                           "shrink": shrink},
                                  missing_color='white',
                                  legend_on=True,
                                  legend_labels=['DJF', 'MAM', 'JJA', 'SON'],
                                  legend_box_xy=legend_box_xy,
                                  legend_box_size=legend_box_size,
                                  legend_lw=legend_lw,
                                  legend_fontsize=legend_fontsize,
                                  logo_rect = logo_rect,
                                  logo_off  = logo_off)

    if landscape:
        ax.set_xticklabels(xaxis_labels, rotation=45, va="bottom", ha="left")
        ax.set_yticklabels(yaxis_labels, rotation=0, va="center", ha="right")
        for xtick, color in zip(ax.get_xticklabels(), lable_colors):
            xtick.set_color(color)
        ax.yaxis.label.set_color(lable_colors[0])
    else:
        ax.set_xticklabels(xaxis_labels, rotation=45, va="bottom", ha="left")
        ax.set_yticklabels(yaxis_labels, rotation=0, va="center", ha="right")
        ax.xaxis.label.set_color(lable_colors[0])
        for ytick, color in zip(ax.get_yticklabels(), lable_colors):
            ytick.set_color(color)

    ax.tick_params(axis="x", labelsize=fontsize)
    ax.tick_params(axis="y", labelsize=fontsize)

    cbar.ax.tick_params(labelsize=fontsize)

    # Add title
    ax.set_title("Model Performance of Seasonal Climatology ({}, {})".format(
                 stat.upper()), fontsize=fontsize*1.5, pad=30)

    # Add Watermark
    if watermark: 
      ax.text(0.5, 0.5, 'E3SM-PCMDI', transform=ax.transAxes,
              fontsize=100, color='black', alpha=0.5,
              ha='center', va='center', rotation=25)
      # Add data info
      fig.text(1.25, 0.9, 'Data version\n'+data_version, transform=ax.transAxes,
               fontsize=12, color='black', alpha=0.6, ha='left', va='top',)

    # Save figure as an image file
    figname = figure_template.replace("%(metric)",stat)
    figfile = os.path.join(outdir,figname)
    fig.savefig(figfile, facecolor='w', bbox_inches='tight')
    del(data_all_nor,xaxis_labels,yaxis_labels,lable_colors)

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

    figsize = (40, 12)
    fontsize = 28
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
    
    var_list  = ['model','run'] + list(enso_lib.columns[2:]) + list(mov_lib.columns[2:]) + list(asl_lib.columns[2:])
    var_names = var_list #[var.split("-")[0] for var in var_list]
    var_units = [var.replace("-"," \n ") for var in var_list]
    merge_tmp = pd.merge(enso_lib,mov_lib,how='inner',on=['model','run'])
    merge_lib = pd.merge(merge_tmp,asl_lib,how='inner',on=['model','run'])
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
    del(data_dict,df_sub1,df_sub2)
    exit()

    ###################################
    #generate portrait plot
    ###################################
    ptrait_fig_dir = os.path.join(outdir,"portrait_4seasons")
    if os.path.exists(ptrait_fig_dir):
      shutil.rmtree(ptrait_fig_dir)
    os.makedirs(ptrait_fig_dir)
    print("Portrait  Plots (4 seasons),loop each region and metric....")
    #########################################################################
    seasons = ['djf', 'mam', 'jja', 'son']
    for metric in merge_dict.keys():
      data_dict = merge_dict[metric]
      var_list = list(merge_dict[metric].columns[3:])
      var_unit_list = []
      for var in var_list:
        strs = var.replace("("," ").replace(")","").split(" ")
        var_unit_list.append("{} \n {}".format(strs[0],strs[1]))
      port4sea_plot(metric,seasons,data_dict,var_list,var_unit_list,
                    cmip_models,test_models,highlight_models,
                    file_template,figure_template,ptrait_fig_dir,
                    data_version=None,watermark=False)

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
  
  parameter.enso_diags = 'EnsoAmpl,EnsoDuration,EnsoSeasonality'
  parameter.enso_base = 'default,default,default'
  parameter.enso_metric = 'metric_value'
  parameter.enso_units = '%,%,%'

# parameter.mov_diags = 'AMO,SAM,PSA1,PSA2'
# parameter.mov_base = 'HadSST2,NOAA-20C,NOAA-20C,NOAA-20C'
# parameter.mov_seasons = 'DJF,JJA,MAM,SON,Monthly'
# parameter.mov_metric = 'frac,rms'
# parameter.mov_units = '%,%,%,%,degC,hPa,hPa,hPa'
  parameter.mov_diags = 'SAM,PSA1,PSA2'
  parameter.mov_base = 'HadSST2,NOAA-20C,NOAA-20C,NOAA-20C'
  parameter.mov_seasons = 'Monthly' #'DJF,JJA,MAM,SON,Monthly'
  parameter.mov_metric = 'frac,rms'
  parameter.mov_units = '%,%,%,hPa,hPa,hPa'
  #parameter.mov_metric = 'rms'
  #parameter.mov_units = 'hPa,hPa,hPa'

  parameter.asl_diags = "lon,lat,ActCenPres"
  parameter.asl_base = 'NOAA-20C'
  parameter.asl_seasons = 'Monthly' #'DJF,JJA,MAM,SON,Monthly'
  parameter.asl_metric = 'rms_xyt'
  parameter.asl_units = 'deg,deg,hPa'

  parameter.output_path = fig_dir
  parameter.ftype = "pdf"
  parameter.debug = False

  main(parameter)

