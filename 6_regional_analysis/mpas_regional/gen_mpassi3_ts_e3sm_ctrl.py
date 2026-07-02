import os
import collections
import xarray as xr
import numpy as np
import pandas as pd
import glob
from datetime import datetime
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from scipy.stats import pearsonr
from scipy.signal import butter, filtfilt, sosfilt,lfilter
from mpl_toolkits.basemap import Basemap
from matplotlib.pylab import rcParams
from matplotlib.patches import Polygon

def generate_yymm(start_year, end_year):
    """Generates a list of YYMM strings between the given years."""
    yymm_list = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            yymm_list.append("{:04d}{:02d}".format(year,month))
    return yymm_list

def main(fig_path,out_path,mip,exp,relm,period,case_id,case_dict,region,ystr,yend): 
  case = list(case_dict.keys())[0]
  var  = case_dict[case]._var
  ftpl = case_dict[case].path
  yymm = []
  if len(ftpl) > 1: 
    for i,ff in enumerate(ftpl):
      if i == 0:
        ds = xr.open_dataset(ff)
      else: 
        ds = xr.merge([ds,xr.open_dataset(ff)],compat='override')
      yymm.append(generate_yymm(ystr,yend))
  else:
    fname = ftpl[0].split("/")[-1]
    ds = xr.open_dataset(ftpl[0])
    yymm  = generate_yymm(int(ystr),int(yend))
    ds = xr.open_dataset(ftpl[0])
  
  times = []
  for ym in yymm:
     times.append('{}-{}-15'.format(ym[0:4],ym[4:6]))
  ds['Time'] = times 
  #select target period 
  ymds = '{}-{}-01'.format(period.split("-")[0][0:4],period.split("-")[0][4:6])
  ymde = '{}-{}-31'.format(period.split("-")[1][0:4],period.split("-")[1][4:6])
  da   = ds.sel(Time=slice(ymds,ymde))
  
  year  = []
  month = []
  for yymm in da['Time'].data: 
    year.append(str(yymm)[0:4])
    month.append(str(yymm)[5:7])
  da['year'] = np.array(year)
  da['month']= np.array(month)

  #construct region list 
  title = ['time','year','month']
  var_list = []
  for var in list(da.keys()):
    if var not in ["startTime", "endTime", "Time", "year", "month"]:
      var_list.append(var)
      title.append(var)
  outdata = pd.DataFrame([],columns=title)
  outdata['time'] = da['Time'].data 
  outdata['year'] = da['year'].data
  outdata['month'] = da['month'].data

  for var in var_list:
    print('working on {}'.format(var))
    outdata[var] = da[var][:].data

  #now,write out data to file
  if not os.path.exists(out_path):
    os.makedirs(out_path)
  out_file = '{}.{}.{}.{}.{}.{}.csv'.format(mip,exp,case,case_id,region,period)
  out_file = os.path.join(out_path,out_file)
  if os.path.exists(out_file):
    os.remove(out_file)
  outdata.to_csv(os.path.join(out_file),index=False)
  
  #plot time series 
  plot_mpas_ts(fig_path,outdata,region,mip,exp,case,case_id,period,var_list) 

  del(ds,da,times,ymds,ymde,title,outdata,out_file)
  
  return

def plot_mpas_ts(fig_path,df,region,mip,exp,case,case_id,period,var_list):
  time   = df['time']
  xtime  = np.linspace(1,len(time),len(time))
  years  = int(time[0].split("-")[0])
  yeare  = int(time[len(time)-1].split("-")[0])
  
  print('plotting {}'.format(region))

  # Plot all timeseries
  xtick = np.arange(0,len(time))
  xlabs = np.arange(0,len(time)) / 12.0  + years
  fontsize = 16
  fig = plt.figure(figsize=(8, 12))
  colors = plt.cm.jet(np.linspace(0,1,len(var_list)))
  for i,var in enumerate(var_list):
    print("plot {}".format(var))
    var0 = np.array(df[var])
    var1 = low_pass(1.0/11.0,var0,axis=0)
    ymin = min(df[var])
    ymax = max(df[var])
    ax = fig.add_subplot(len(var_list), 1, i+1)
    ax.plot(xtime, var0, color='grey', alpha=1.0,  linewidth=0.8, label='monthly')
    ax.plot(xtime, var1, color='black', alpha=1.0, linewidth=1.4, label='11-point Hamming')
    #ax.fill_between(xtime, 0., ssta_series, ssta_series> 0., color='red',   alpha=.75)
    #ax.fill_between(xtime, 0., ssta_series, ssta_series< 0., color='blue',  alpha=.75)
    #ax1.xaxis.tick_top()    
    #ax.set_title('Time series of monthly mean ASL longitude index (v3)')
    #ax.legend(['5-month running mean'])
    ax.set_xlabel('Time (years)',fontsize=fontsize*1.1)
    ax.set_ylabel("{}".format(var),fontsize=fontsize*1.1)
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(0,len(time))
    ax.set_xticks(xtick[::120])
    ax.set_xticklabels(xlabs[::120].astype(int))
    ax.tick_params(labelsize=fontsize)
    ax.grid(True)
    ax.set_title("{} time series".format(var), fontsize=fontsize*1.1)
    if i+1 == len(var_list):
        ax.legend(loc='lower right', prop={'size':8})
    del(var0,var1)
  #plt.suptitle('Ocean Regional Mean Timeseries', fontsize=fontsize*1.1)

  # Adding legend, x and y labels, and titles for the lines
  #plt.legend()

  #plt.draw()
  plt.tight_layout()

  if not os.path.exists(fig_path):
    os.makedirs(fig_path)
  fig_out = os.path.join(fig_path, "fig_{}_{}_{}_{}_{}_{}.pdf".format(region,mip,exp,case,case_id,period))
  print(fig_out)
  plt.savefig(fig_out)
  plt.close()

  return 

def low_pass(cutoff_freq, data, order=5, axis=-1):
    #low-pass: filtering high-frequency signal
    #nyquist normalized cutoff for digital design
    Wn = cutoff_freq
    b, a = butter(order, Wn, btype='lowpass', analog=False)
    data_filt = filtfilt(b, a, data, axis=axis, method="gust")
    return data_filt

class Case:
  def __init__(self, path, var, color, label):
      self._path = path
      self._color = color
      self._label = label
      self._var = var

      return

  @property
  def path(self):
      return self._path

  @property
  def color(self):
      return self._color

  @property
  def label(self):
      return self._label

  @property
  def label(var):
      return self._var

if __name__ == "__main__":

  top_path = "/lcrc/group/e3sm/ac.szhang/acme_scratch/e3sm_project"
  out_path = os.path.join(top_path,"E3SMv21_testings","paper_material","fig_data","mpas_ts","raw_index")
  fig_path = os.path.join(top_path,"E3SMv21_testings","paper_material","6_regional_analysis","mpas_regional","figure")

  # region of interest
  region   = "seaIceAreaVolSH"
  mip      = "e3sm"
  exps     = ["piControl"]
  relms    = ["1950"]
  product  = "v2_1-SORRM"
  case_id  = 'mpas_ts'
  period   = "080101-100012"
  run_path = "/lcrc/group/acme/ac.dcomeau/scratch/chrys/%(EXPS)/mpas_analysis_output/%(PERIOD)/timeseries"
  for exp in exps:
    for relm in relms:
      filePath = []
      ystr = '0701' #period.split("-")[0][0:4]
      yend = '1000' #period.split("-")[1][0:4]
     #rnam ='{}.{}.{}_{}'.format(product.split("-")[0],product.split("-")[1],exp,relm)
      rnam = '20221116.CRYO1950.ne30pg2_SOwISC12to60E2r4.N2Dependent.submeso.chrysalis'
      ptmp = run_path.replace("%(EXPS)",rnam).replace("%(PERIOD)",'yrs951-1000')
      filePath.append(os.path.join(ptmp,'{}.nc'.format(region)))
      print(ptmp)
      case = '{}.{}'.format(product,relm)
      case_dict = collections.OrderedDict()
      case_dict[case] = Case(filePath,var=region,color="blue",label=case)
      #call fuction to generate nino index
      main(fig_path,out_path,mip,exp,relm,period,case_id,case_dict,region,ystr,yend)
      del(case,case_dict,filePath)
