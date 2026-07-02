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

def load_metric_file_list(path,mip,exp,ver): 
    file_dict = {}
    ftest = glob.glob(os.path.join(path,"{}.{}.{}.{}.{}.csv".format(mip,exp,"*",ver,"*")))
    for ff in ftest: 
      if os.path.exists(ff):
        fname = ff.split("/")[-1]
        product = fname.split(".")[2]
        relm = fname.split(".")[3]
        #print(mip,exp,product,relm)
        if mip == "obs":
          product = product.replace("_","-")
        if product not in file_dict.keys():
          file_dict[product] = {}
        if product not in file_dict[product].keys():
          file_dict[product][relm] = {}
        file_dict[product][relm] = ff 
    del (ftest)
    return file_dict 

def load_and_process_data(file_dict,seasons,indices,out_path):
    #reorgnize the data and derive seaonal mean and annual cycle climatology
    for mip in file_dict.keys(): 
      for exp in file_dict[mip].keys():
        for prod in file_dict[mip][exp].keys():
          for relm in file_dict[mip][exp][prod].keys():
             csvObj = file_dict[mip][exp][prod][relm]
             data = pd.read_csv(csvObj)
             data = data.reset_index(names=['year']) #(inplace=True)
             data['year'] = pd.DatetimeIndex(data['time']).year
             data['month'] = pd.DatetimeIndex(data['time']).month
             data['day'] = pd.DatetimeIndex(data['time']).day
             data = data.drop(columns=['time'])
             for sea in seasons: 
               print("mip,exp,prod,relm=",mip,exp,prod,relm,sea)  
               if sea in ['DJF']: 
                 sub_data  = data[data['month'].isin([12,1,2])].copy()
                 datmn = sub_data.groupby(by=['year'],dropna=True).mean()
               if sea in ['JJA']:
                 sub_data  = data[data['month'].isin([6,7,8])].copy()
                 datmn = sub_data.groupby(by=['year'],dropna=True).mean()
               if sea in ['MAM']:
                 sub_data  = data[data['month'].isin([3,4,5])].copy()
                 datmn = sub_data.groupby(by=['year'],dropna=True).mean()
               if sea in ['SON']:
                 sub_data  = data[data['month'].isin([9,10,11])].copy()
                 datmn = sub_data.groupby(by=['year'],dropna=True).mean()
               if sea in ['ANN']:
                 sub_data  = data
                 datmn = sub_data.groupby(by=['year'],dropna=True).mean()
               if sea in ['AC']:
                 sub_data  = data
                 datmn = sub_data.groupby(by=['month'],dropna=True).mean()
                 print(datmn)
               if sea in ['Monthly']:
                 sub_data  = data
                 datmn = data 
               out_dir = os.path.join(out_path,mip,exp)
               if not os.path.exists(out_dir):
                  os.makedirs(out_dir)
               period = '{:04d}-{:04d}'.format(min(data['year']),max(data['year'])) 
               out_file = 'ASL.index.{}.{}.{}.{}.csv'.format(prod,relm,sea,period)
               out_file = os.path.join(out_dir,out_file) 
               if os.path.exists(out_file):
                 os.remove(out_file)
               print(out_dir,out_file)
               if sea in ['Monthly']:
                 month = datmn['month']
                 day = datmn['day']
                 datmn = datmn.drop(columns=['month'])
                 datmn = datmn.drop(columns=['day'])
                 outdata = pd.DataFrame(datmn)
                 outdata.insert(1,"month",month)
                 outdata.insert(2,"day",day)
                 outdata.to_csv(os.path.join(out_file),index=False)
                 del(month,day)
               elif sea in ['DJF',"JJA","MAM","SON","ANN"]:
                 year = datmn['month']
                 year = np.arange(int(period.split("-")[0]),int(period.split("-")[1])+1)
                 datmn = datmn.drop(columns=['month'])
                 datmn = datmn.drop(columns=['day'])
                 outdata = pd.DataFrame(datmn)
                 outdata.insert(0,"year",year)
                 outdata.to_csv(os.path.join(out_file),index=False)
                 del(year)
               else:
                 month = datmn['year']
                 month = np.arange(1,13,1)
                 datmn = datmn.drop(columns=['year'])
                 outdata = pd.DataFrame(datmn)
                 outdata.insert(0,"month",month) 
                 outdata.to_csv(os.path.join(out_file),index=False) 
                 del(month)
               del(sub_data,datmn,outdata)    
             del(data,csvObj)
    return 

def main(mips,exps,ver,seasons,indices,data_path,out_path):
    # identify data in .csv file
    file_dict = {}
    for mip in mips: 
      for exp in exps: 
        # find list of metrics data files
        if mip not in file_dict.keys():
          file_dict[mip] = {}
        if exp not in file_dict[mip].keys():  
          file_dict[mip][exp] = {}
        file_dict[mip][exp] = load_metric_file_list(data_path,mip,exp,ver)
         
    load_and_process_data(file_dict,seasons,indices,out_path)

    return

if __name__ == "__main__":
  data_dir = "/lcrc/group/e3sm/ac.szhang/acme_scratch/e3sm_project/E3SMv21_testings/paper_material"
  data_path = os.path.join(data_dir,"fig_data","asl_analysis","raw_index")
  out_path = os.path.join(data_dir,"fig_data","asl_analysis","ts_index")
  mips = ["analysis","cmip6","e3sm"]
  exps = ["historical","ssp370"]
  ver  = "asl_scotthoskingv3" 
  seasons = ['ANN','DJF','JJA','MAM','SON','AC','Monthly'] 
  indices = ['lon','lat','ActCenPres','SectorPres','RelCenPres']
  
  main(mips,exps,ver,seasons,indices,data_path,out_path)

