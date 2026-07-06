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

def main(mode,exps,seas,dir_hist,dir_sspf,period_hist,period_sspf,out_dir):
    # density of reference seawater (kg/m^3)
    rho_sw = 1026.
    
    # Heat capacity of seawater
    cp_sw  = 3.996e3
    
    #low pass to remove fluctuations less than 24 months (2 years).
    sample_time = 1 # unit: month
    cutoff_freq = 1.0/5.0 # 5-year filtering  

    for sea in seas: 
      labels = []
      eof_hst = []
      eof_ssp = []
      var_frc = []
      dts_raw = []
      dts_flt = []
      for exp in exps:
        ekey = exp.split(".")  
        if 'FISMF' in exp:
          fil1 = glob.glob(os.path.join(dir_hist,
                 '{}_{}_*_{}_*_{}_{}_*cbf.nc'.format(mode,sea,ekey[0].replace('-FISMF',''),ekey[1],period_hist)))
          fil2 = glob.glob(os.path.join(dir_sspf,
                 '{}_{}_*_{}_*_{}_{}_*cbf.nc'.format(mode,sea,ekey[0],ekey[1],period_sspf)))
        else:
          fil1 = glob.glob(os.path.join(dir_hist,
                 '{}_{}_*_{}_*_{}_{}_*cbf.nc'.format(mode,sea,ekey[0],ekey[1],period_hist)))
          fil2 = glob.glob(os.path.join(dir_sspf,
                 '{}_{}_*_{}_*_{}_{}_*cbf.nc'.format(mode,sea,ekey[0],ekey[1],period_sspf)))
        f1   = xr.open_mfdataset(fil1)
        f2   = xr.open_mfdataset(fil2)
        print(fil1)
        print(fil2)
        pc1  = f1['pc']
        pc2  = f2['pc']
        eof1 = f1['eof']
        eof2 = f2['eof']
        fra1 = f1['frac'].data * 100.0
        fra2 = f2['frac'].data * 100.0
        pcs  = xr.concat([pc1, pc2], dim="time") 
        
        #Extract the data
        dtmp_raw = np.array(pcs.data).ravel()
        dtmp_flt = low_pass(cutoff_freq,sample_time,dtmp_raw)
        dts_raw.append(dtmp_raw)
        dts_flt.append(dtmp_flt)
        labels.append("{}-{}".format(ekey[0],ekey[1]))
        del(ekey,fil1,fil2,f1,f2,pc1,pc2,eof1,eof2,fra1,fra2,pcs,dtmp_raw,dtmp_flt)

      # plot
      plot_time_series(mode,sea,labels,period_hist,period_sspf,dts_raw,dts_flt,out_dir)
      del(dts_raw,dts_flt,labels)

def plot_time_series(mode,sea,labels,period_hist,period_sspf,data,dflt,out_dir):
  #color map for markers
  title  = "{} PC Time Series ({})".format(mode.split("_")[0].upper(),sea.upper())
  ylabel = 'PC ({})'.format(mode.split("_")[2].upper())
  xlabel = 'Model Time (year)'
  colormap = "tab20_r"
  fontsize = 18
  #color map for highlight lines
  lncolors = ['#377eb8', '#e41a1c', '#ff7f00', '#4daf4a',
              '#f781bf','#a65628', '#984ea3','#999999', '#dede00']
  lnthicks = [1,1,1,1,1,1,1,1,1,1,1]
  lnlstyle = ['solid','solid','solid','solid','solid',
              'solid','solid','solid','solid','solid','solid',]

  #totoal period for time series plot    
  period = "{}-{}".format(period_hist.split("-")[0], period_sspf.split("-")[1])
  
  #convert data to numpy array
  data = np.array(data)
  dflt = np.array(dflt)
  print(dflt.min(), dflt.max())
  print(data.shape, dflt.shape)

  #ensemble mean
  dfmn = dflt.mean(axis=0)
  reflabels = ['v2_1-SORRM (Mean)']
  lrcolors = ['#000000','#999999']
  lrthicks = [3,3]
  lrlstyle = ['solid','solid']

  years = int(period_hist.split("-")[0])
  yeare = int(period_sspf.split("-")[1])
  yearm = int(period_hist.split("-")[1])
  nyear = yeare - years + 1
  nmons = nyear * 12
  xref  = np.array([yearm,yearm+25,yearm+45])
  xref  = np.array([2010,yearm+25,yearm+45])
  xref  = xref - years

  vmax = data.max() #dflt.max()
  vmin = data.min() #dflt.min()
  vmax = vmax + max(abs(vmax),abs(vmin))/50.0
  vmin = vmin - max(abs(vmax),abs(vmin))/50.0

  vmin = -5
  vmax = 5

  plt.figure(figsize = (12,8))
  plt.rcParams.update({'font.size': fontsize})
  ax = plt.axes() 

  #plt.subplot(1, 1, 2)
  poplgd = []
  lablgd = []
  labcol = []
  labthk = []
  lablst = []
  for i,exp in enumerate(exps):
    #plt.plot(data[i,:], color=lncolors[i], linewidth=lnthicks[i], alpha=0.2, label=labels[i])  
    #plt.plot(dflt[i,:], color=lncolors[i], linewidth=lnthicks[i]*3.0, alpha=1.0, label=labels[i])
    plt.plot(data[i,:], color=lncolors[i], linewidth=lnthicks[i], alpha=0.2)
    plt.plot(dflt[i,:], color=lncolors[i], linewidth=lnthicks[i]*3.0, alpha=1.0)
    #create lengend with color box
    #poplgd.append(mpatches.Patch(color=lncolors[i],linewidth=lnthicks[i]*0.1, label=labels[i]))
    #poplgd.append(mpatches.Patch(color=lncolors[i],linewidth=lnthicks[i]*0.3, label=labels[i]))
    #legend labels
    #lablgd.append(labels[i]) 
    #labcol.append(lncolors[i])
    #labthk.append(lnthicks[i])  
    #lablst.append(lnlstyle[i])
    lablgd.append(labels[i]+"(Filter)")
    labcol.append(lncolors[i])
    labthk.append(lnthicks[i]*3.0)
    lablst.append(lnlstyle[i]) 

  #plot lines for ensemble mean
  plt.plot(dfmn, color=lrcolors[0], linewidth=lrthicks[0]*2.0, linestyle=lrlstyle[0],alpha=1.0)
  lablgd.append(reflabels[0])
  labcol.append(lrcolors[0])
  labthk.append(lrthicks[0]*2.0)
  lablst.append(lrlstyle[0])

  #add legend 
  lines   = [Line2D([0], [0], color=labcol[i], linewidth=labthk[i], linestyle=lnlstyle[i]) 
             for i in range(len(lablgd))]
  plt.legend(lines, lablgd, fontsize=fontsize*1.1, loc="upper right")

  #plot grid and axis
  #plt.grid(True, which='both',color = 'grey', linestyle = '--', linewidth = 0.2)
  plt.xlim(-0.5,nyear+0.5)
  plt.ylim(vmin,vmax)
  
  #plot reference line 
  plt.axhline(y = 0.0, color = 'grey', linestyle = 'dashed', linewidth=1.5)
  for i,ref in enumerate(xref):
     if i == 0:
       lwidth = 2.0
       lcolor = "black"
     else:
       lwidth = 1.5
       lcolor = "black"
     plt.axvline(x=ref,  color=lcolor, linestyle = 'dashed', linewidth=lwidth)

  #ax.xaxis.tick_top()
  ax.set_title(title,   fontsize=fontsize*1.1)
  ax.set_xlabel(xlabel, fontsize=fontsize*1.1)
  ax.set_ylabel(ylabel, fontsize=fontsize*1.1)

  # setting ticks for x-axis
  xticks = np.arange(1,nyear, 20)
  xlabls = np.arange(years,yeare,20)
  ax.set_xticks(xticks)
  ax.set_xticklabels(xlabls)
  ax.tick_params(labelsize=fontsize)

  #plt.show()
  figname = os.path.join(out_dir,'fig_{}_pcs_{}_{}.pdf'.format(mode,sea,period))
  plt.savefig(figname)
  
  return 

if __name__ == "__main__":
  results_dir = "/lcrc/group/e3sm/ac.szhang/acme_scratch/e3sm_project/E3SMv21_testings/paper_material"
  data_dir = "/lcrc/group/e3sm/public_html/diagnostic_output/ac.szhang/E3SMv2_1/pcmdi"
  ts_path = os.path.join(data_dir,"diagnostic_results","variability_modes")
  fig_dir = os.path.join(results_dir,"4_sam_analysis","sam_index_ts","figure")

  exps = ["v2_1-SORRM.0701","v2_1-SORRM.0751","v2_1-SORRM.0801",'v2_1-SORRM-FISMF.0701']
  seas = ["DJF","JJA","SON","MAM"]
  mode = "SAM_PSL_EOF1"
  dir_hist = os.path.join(ts_path,"e3sm/historical/v20241111/SAM/NOAA-20C")
  dir_sspf = os.path.join(ts_path,"e3sm/ssp370/v20241111/SAM/NOAA-20C")
  period_hist = "1950-2014"
  period_sspf = "2015-2100"
  main(mode,exps,seas,dir_hist,dir_sspf,period_hist,period_sspf,fig_dir)
