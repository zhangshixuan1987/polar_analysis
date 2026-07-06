import shutil
import gc
import glob
import json
import os
import sys
import time
import cdms2
import pandas as pd
import netCDF4
import xarray as xr
import numpy as np
from os import path
from numpy.fft import rfft
from scipy.signal import butter, filtfilt, sosfilt,lfilter
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from collections import OrderedDict

def low_pass(cutoff_freq, sample_time, data, order=5, l_check_filter=False):
    #low-pass: filtering high-frequency signal 
    sample_rate = 1.0 / sample_time
    #highest frequency of variation that the observed data can provide any variation
    nyquist_freq = 0.5 * sample_rate
    # nyquist normalized cutoff for digital design
    Wn = cutoff_freq #/ nyquist_freq
    print(cutoff_freq,nyquist_freq)
    b, a = butter(order, Wn, btype='lowpass', analog=False)
    if l_check_filter == True: 
      check_filtering(b,a)    
    data_filt = filtfilt(b, a, data, method="gust")
    return data_filt

def high_pass(cutoff_freq, sample_time, data, order=5, l_check_filter=False):
    #high-pass: filtering low-frequency signal
    sample_rate = 1.0 / sample_time
    #highest frequency of variation that the observed data can provide any variation
    nyquist_freq = 0.5 * sample_rate
    # nyquist normalized cutoff for digital design
    Wn = cutoff_freq #/ nyquist_freq
    b, a = butter(order, Wn, btype='highpass', analog=False)
    if l_check_filter == True: 
      check_filtering(b,a)
    data_filt = filtfilt(b, a, data, method="gust")
    return data_filt

def band_pass(cutoff_freq, sample_time, data, order=5, l_check_filter=False):
    #band_pass: keep the single within the band 
    sample_rate = 1.0 / sample_time
    #highest frequency of variation that the observed data can provide any variation
    nyquist_freq = 0.5 * sample_rate
    Wn = cutoff_freq #/ nyquist_freq
    b, a = butter(order, Wn, btype='band', analog=False)
    if l_check_filter == True: 
      check_filtering(b,a)
    data_filt = filtfilt(b, a, data, method="gust")
    return data_filt

def check_filtering(w,h):
    #check the filtering behavior
    w, h = signal.freqs(b, a)
    plt.semilogx(w, 20 * np.log10(abs(h)))
    plt.semilogx(20*np.log10(np.abs(rfft(dflt[0,:]))))
    plt.ylim(-100, 20)
    plt.title('Butterworth filter frequency response')
    plt.xlabel('Frequency [radians / second]')
    plt.ylabel('Amplitude [dB]')
    plt.margins(0, 0.1)
    plt.grid(True, which='both',axis='both')
    plt.axvline(wn, color='green') # cutoff frequency
    plt.show()
    return 

def plot_annual_cycle(metrics,metunit,relms,products,periods,data_dict,fig_dir):
  #color map for markers
  colormap = "tab20_r"
  fontsize = 18
  #color map for highlight lines
  lncolors = ['#000000', '#ff7f00', '#4daf4a', '#377eb8', '#e41a1c', 
              '#f781bf','#a65628', '#984ea3','#999999', '#dede00']
  lnthicks = [1,1,1,1,1,1,1,1,1,1,1]
  lnlstyle = ['solid','solid','solid','solid','solid',
              'solid','solid','solid','solid','solid','solid',]

  xtick = np.arange(0,12,1)
  #xlabs = ['J','F','M','A','M','J','J','A','S','O','N','D']
  xlabs = ['M','A','M','J','J','A','S','O','N','D','J','F']

  vmin = [218,-72,965,-12] 
  vmax = [258,-64,985,-6]
  fontsize = 14
  plt.rcParams.update({'font.size': fontsize})
  #fig = plt.figure(figsize=(8, 8))
  fig, ax = plt.subplots(2, 2,figsize=(12, 8))
  for k,met in enumerate(metrics): 
    metunt = metunit[k]
    ylabel = '{} ({})'.format(met.upper(),metunt)
    xlabel = 'Annual cycle'
    title  = "{} (ASL) Time Series".format(met)
    #ax = fig.add_subplot(2, 2, k+1)
    for i,period in enumerate(periods): 
      meanst = []
      stdvst = []
      for prod in products: 
        tmp = []
        for rel in relms:
          if period in data_dict[prod][rel].keys():  
            tmp.append(data_dict[prod][rel][period]['mean'][met])
        mean = np.roll(np.array(tmp).mean(axis=0),-2, axis=0) 
        stdv = np.roll(np.array(tmp).std(axis=0),-2, axis=0) 
        meanst.append(mean)
        stdvst.append(stdv)
        del(tmp,mean,stdv)
      ax.flat[k].plot(xtick,meanst[0][:],color=lncolors[i],alpha=1.0,linewidth=lnthicks[i]*2.0,linestyle=lnlstyle[i])
      #ax.flat[k].plot(xtick,meanst[1][:],color=lncolors[i],alpha=1.0,linewidth=lnthicks[i],linestyle='dashed')
      ax.flat[k].fill_between(xtick,meanst[0][:]-stdvst[0][:],meanst[0][:]+stdvst[0][:],alpha=0.2,facecolor=lncolors[i])
      del(meanst,stdvst) 
    #plot reference line 
    #plt.axhline(y = 0.0, color = 'grey', linestyle = 'dashed', linewidth=1.5)
    #for i,ref in enumerate(xref):
    #   if i == 0: 
    #     lwidth = 2.0
    #     lcolor = "black"
    #   else:
    #     lwidth = 1.5
    #     lcolor = "black"
    #   plt.axvline(x=ref,  color=lcolor, linestyle = 'dashed', linewidth=lwidth) 

    #add legend
    lines   = [Line2D([0],[0],color=lncolors[i],linewidth=lnthicks[i]*2.0,linestyle=lnlstyle[i])
               for i in range(len(periods))]
    if k == 1: 
      #plt.legend(lines,periods,fontsize=fontsize*1.1,loc="center right")
      #plt.legend(lines,periods,fontsize=fontsize*1.1,loc="lower left")
      ax.flat[k].legend(lines,periods,fontsize=fontsize*1.1,loc="upper left")

    #ax.xaxis.tick_top()
    ax.flat[k].set_title(title,   fontsize=fontsize*1.1)
   #ax.flat[k].set_xlabel(xlabel, fontsize=fontsize*1.1)
    ax.flat[k].set_ylabel(ylabel, fontsize=fontsize*1.1)

    #plt.grid(True, which='both',color = 'grey', linestyle = '--', linewidth = 0.2)
    ax.flat[k].set_xlim(-0.5,len(xtick)-0.5)
    ax.flat[k].set_xticks(xtick)
    ax.flat[k].set_xticklabels(xlabs)
    ax.flat[k].set_ylim(vmin[k],vmax[k])

    # setting ticks for x-axis
    ax.flat[k].tick_params(labelsize=fontsize)
    ax.flat[k].grid(False)
  
  #plt.draw()
  plt.tight_layout()

  #plt.show()
  if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)  
  figname = os.path.join(fig_dir,'fig_asl_annual_cycle_hist_vs_ssp370_.pdf') 
  plt.savefig(figname)
  
  return 

if __name__ == "__main__":
  top_dir = "/lcrc/group/e3sm/public_html/diagnostic_output/ac.szhang/polar_analysis/data"
  data_dir = os.path.join(top_dir,"fig_data","asl_analysis","raw_index")
  fig_dir = os.path.join("/lcrc/group/e3sm/public_html/diagnostic_output/ac.szhang/polar_analysis/figure", "asl_analysis", "metrics_plot")
  
  mip = "e3sm" 
  ver = "asl_scotthoskingv3.PSL" 
  exp = "historical_ssp370"
  runs = ['v2_1-SORRM.0701','v2_1-SORRM.0751','v2_1-SORRM.0801','v2_1-SORRM-FISMF.0751']
  periods = ['1951-1980','1981-2010','2011-2040','2041-2070','2071-2100']
  metrics = ['lon',   'lat',   'ActCenPres','RelCenPres']
  metunit = ['degree','degree','hPa',       'hPa'       ]
  #main(ver,metrics,metunit,mip,exp,relms,products,periods,data_dir,fig_dir)

  #extract data
  data_dict = {}
  for run in runs: 
     prod = run.split(".")[0]
     rel = run.split(".")[1]
     if prod not in data_dict.keys():
       data_dict[prod] = {}
     if rel not in data_dict[prod].keys():
       data_dict[prod][rel] = {}
     print(prod,rel,prod,rel,ver)

     fc  = '{}.{}.{}.{}.{}.195001-201412.csv'.format(mip,exp.split("_")[0],prod,rel,ver)
     ff  = '{}.{}.{}.{}.{}.201501-210012.csv'.format(mip,exp.split("_")[1],prod,rel,ver)
     if 'FISMF' in prod: 
       fc = fc.replace('-FISMF','')
     
     fcl = os.path.join(data_dir,fc)
     ffl = os.path.join(data_dir,ff)
     if os.path.exists(fcl) and os.path.exists(ffl):
       f1 = pd.read_csv(fcl,index_col=False)
       f2 = pd.read_csv(ffl,index_col=False)
       df = pd.concat([f1, f2], join="inner")
       print(df.columns)
       df['year'] = pd.to_datetime(df['time']).dt.year
       df['month'] = pd.to_datetime(df['time']).dt.month
       df = df.drop(columns=['time'])
       for period in periods:
         ystr = int(period.split("-")[0])
         yend = int(period.split("-")[1])
         dsub = df[(df["year"] >= ystr) & (df["year"] <= yend)]
         if period not in data_dict[prod][rel].keys():
           data_dict[prod][rel][period] = {}
         data_dict[prod][rel][period]['mean'] = dsub.groupby(by=["month"], dropna=False).mean()
         data_dict[prod][rel][period]['stdev'] = dsub.groupby(by=["month"], dropna=False).std()
         del(dsub,ystr,yend)
       del(f1,f2,df)
     del[fc,ff,fcl,ffl]
  # plot time series 
  #plot_annual_cycle(metrics,metunit,relms,products,periods,data_dict,fig_dir)

