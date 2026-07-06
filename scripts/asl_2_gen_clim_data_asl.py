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
from math import isnan
from collections import OrderedDict
from re import split
from pcmdi_metrics import stats
from pcmdi_metrics.graphics import Metrics
from pcmdi_metrics.graphics import normalize_by_median
from pcmdi_metrics.graphics import portrait_plot
from pcmdi_metrics.graphics import parallel_coordinate_plot

def load_metric_file_list(path,mip,exp,sea): 
    file_dict = {}
    ftest = glob.glob(os.path.join(path,mip,exp,"ASL.index.{}.{}.{}.csv".format("*",sea,"*")))
    for ff in ftest: 
      if os.path.exists(ff):
        fname = ff.split("/")[-1]
        product = fname.split(".")[2]
        relm = fname.split(".")[3]
        if product not in file_dict.keys():
          file_dict[product] = {}
        if product not in file_dict[product].keys():
          file_dict[product][relm] = {}
        file_dict[product][relm] = ff 
    del (ftest)
    return file_dict 

def process_mean_climatology(ref_data,test_data,indices,sea,period):
    metrics = ['mean','mean_obs','std','std_obs','bias','std_xyt','rms_xyt']
    metric_lib = {}
    for met in metrics: 
      for var in indices:
        if met not in metric_lib.keys():
          metric_lib[met] = {}  
        if var not in metric_lib[met].keys():
          metric_lib[met][var] = {}
    #reorgnize the data and derive seaonal mean and annual cycle climatology
    syear = int(period.split("-")[0])
    eyear = int(period.split("-")[1])
    mask1 = ref_data['year'].between(syear,eyear)
    x1 = ref_data[mask1]
    syear1 = min(x1['year'])
    eyear1 = max(x1['year'])
    mask2 = test_data['year'].between(syear1,eyear1)
    y1 = test_data[mask2]
    if sea in ['AC']: 
      x = x1.groupby(by=['month'],dropna=True).mean() 
      y = y1.groupby(by=['month'],dropna=True).mean()
    else:
      x = x1
      y = y1 

    for var in indices: 
      x0 = x[var].to_numpy().astype(np.float32)
      y0 = y[var].to_numpy().astype(np.float32)
      df = y0 - x0 
      ds = df * df 
      metric_lib['mean'][var]     = np.nanmean(y0, axis=0) 
      metric_lib['mean_obs'][var] = np.nanmean(x0, axis=0)
      metric_lib['std'][var]      = np.nanstd(y0, axis=0) 
      metric_lib['std_obs'][var]  = np.nanstd(x0, axis=0) 
      metric_lib['bias'][var]     = np.nanmean(df, axis=0)
      metric_lib['std_xyt'][var]  = np.nanstd(df, axis=0)
      metric_lib['rms_xyt'][var]  = np.sqrt(np.nanmean(ds))
      del(x0,y0,df,ds)
    del(x1,y1,x,y,syear,eyear,syear1,eyear1,mask1,mask2)
    return metric_lib

def main(obs_sets,obs_mips,test_mips,test_exps,periods,seasons,indices,data_path,out_path):
    # identify data in .csv file
    for ii,obs in enumerate(obs_sets): 
      for sea in seasons: 
        if sea == "AC":
          season_tag = "Monthly"
        else:
          season_tag = sea 
        ofils = glob.glob(os.path.join(data_path,obs_mips[ii],'historical',
                                       "ASL.index.{}.{}.*.csv".format(obs,season_tag)))
        if len(ofils) < 1: 
          exit("no observational file, aborting ...")
        if len(ofils) > 1: 
          print("multiple files for {} show up, select the first file to process".format(obs)) 
        out_lib = {}
        period = periods[ii]
        ref_data = pd.read_csv(ofils[0],index_col=False)
        for mip in test_mips:
          for exp in test_exps:
            test_file = load_metric_file_list(data_path,mip,exp,season_tag)
            for prod in test_file.keys():
              for relm in test_file[prod].keys():
                test_data = pd.read_csv(test_file[prod][relm],index_col=False) 
                metric_lib = process_mean_climatology(
                                   ref_data,test_data,indices,sea,period
                                   )
                for metric in metric_lib.keys():
                  dtmp = []
                  dtmp.append(mip)
                  dtmp.append(exp)
                  dtmp.append(prod)
                  dtmp.append(relm)
                  dtmp.append('{}_{}'.format(prod,relm))
                  title = ['mip','exp','model','run','model_run']
                  for var in metric_lib[metric].keys():
                    title.append(var)  
                    dtmp.append(metric_lib[metric][var])
                  if metric not in out_lib.keys():
                    out_lib[metric] = {}
                    out_lib[metric] = pd.DataFrame([dtmp],columns=title)
                  else:
                    out_lib[metric] = pd.concat(
                            [out_lib[metric],pd.DataFrame([dtmp],columns=title)],
                            ignore_index=True)  
                  del(dtmp)
                del(test_data,metric_lib) 
                
        #now,write out data to file       
        out_dir = os.path.join(out_path,obs)
        if not os.path.exists(out_dir):
           os.makedirs(out_dir)
        for metric in out_lib.keys():
          out_file = 'ASL.{}.clim.{}.{}.csv'.format(metric,sea,period)
          out_file = os.path.join(out_dir,out_file)
          if os.path.exists(out_file):
            os.remove(out_file)  
          outdata = out_lib[metric]
          outdata = outdata.drop(columns=['mip'])
          outdata = outdata.drop(columns=['exp'])
          outdata.to_csv(os.path.join(out_file),index=False)
          del(outdata)
        del(out_lib,ref_data)  
    return

if __name__ == "__main__":
  data_dir = "/lcrc/group/e3sm/public_html/diagnostic_output/ac.szhang/polar_analysis/data"
  data_path = os.path.join(data_dir,"fig_data","asl_analysis","ts_index")
  out_path  = os.path.join(data_dir,"fig_data","asl_analysis","clim_index")
  obs_sets  = ['NOAA_20C.en00','ERA5.en00']
  obs_mips  = ['analysis','analysis']
  periods   = ["1950-2014", "1979-2014" ]
  test_mips = ["cmip6","e3sm"]
  test_exps = ["historical"]
  seasons   = ['ANN','DJF','JJA','MAM','SON','AC'] 
  indices   = ['lon','lat','ActCenPres','SectorPres','RelCenPres']
  
  main(obs_sets,obs_mips,test_mips,test_exps,periods,seasons,indices,data_path,out_path)

