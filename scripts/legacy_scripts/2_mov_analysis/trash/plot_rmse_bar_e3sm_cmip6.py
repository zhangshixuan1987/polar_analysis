import os
import sys
import glob
import json
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import matplotlib.transforms as mtransforms
import numpy as np
import numpy.ma as ma
import collections
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime
from skimage.feature import peak_local_max
import cartopy.crs as ccrs

def main(exps,labels,run_path,fig_path):
  # --- Main ---

  # variables 
  variables = get_varible()
  print(variables)
  # Seasons
  seasons = ['ANN', 'DJF', 'MAM', 'JJA', 'SON']

  # -----------------------------------------------------------------------------
  # Create plot: first only with CMIP6, E3SMv1 and v2
  fig = plt.figure(figsize=[12,9])
  nsx = 4
  nsy = 3

  cmip6 = read_e3sm_diags_metrics(run_path, variables, seasons)
  print(cmip6)
  exit()

  nmodels = len(cmip6['models'])
  nvariables = len(variables)
  nseasons = len(seasons)
  #for ivariable in range(nvariables):

  print("working on ",case,var)

  dsm  = xr.open_dataset(dmsk)
  ds   = xr.open_dataset(data)
  if len(ds.dims) < 3: 
    print("data dimension is incorrect")
    exit()
  else:
    if ds[var].dims[1] == "lat" or ds[var].dims[2] == "lon":
      ds = ds.rename({ds[var].dims[1] : 'latitude',
                      ds[var].dims[2] : 'longitude'})
    if dsm[vmsk].dims[0] == "lat" or dsm[vmsk].dims[1] == "lon":
      dsm = dsm.rename({dsm[vmsk].dims[0] : 'latitude',
                        dsm[vmsk].dims[1] : 'longitude'})
  mask = dsm[vmsk]
  mask = mask /100.0 #range[0,1]
  #print(mask.min(),mask.max())

  #extract pressure data  
  da   = ds[var]
  if da.units == "Pa":
    # change units
    print("change units: from ", da.units, " to ", "hPa")
    da = da / 100. 
    da = da.assign_attrs(units='hPa')

  ################################
  #sanity check on asl region
  ################################
  if l_check_asl_region:
    #only select first year to check  
    da_t = da.sel(time=da.time[0:11]) #.dt.year.values[0])
    #Select region and apply land-sea mask
    da_mask = da_t.where(mask == 0)
    ### slice area around ASL region
    da_nmsk = slice_region(da_t, asl_region)
    da_mask = slice_region(da_mask, asl_region)
    print(da_mask)
    print("before mask,min/max: ", da.min(),da.max())
    print("after mask,min/max: ",da_mask.min(),da_mask.max())
    plot_regions_mask(fig_path,da_nmsk,da_mask,case)
    del(da_t,da_mask,da_nmsk)
  ################################
  #end sanity check on asl region
  ################################

  #start process ASL index
  #Loop through each month and identify lows 
  #years = list(da.time.dt.year.data)
  times = da.time.dt.strftime("%Y-%m-%d") 
  ntime = len(times) 
  all_lows_dfs = pd.DataFrame() 
  print("number of total months in data: ", ntime)
  for t in range(ntime):
    #Select time to process 
    da_t = da.isel(time=t)
    #Apply land-sea mask 
    da_mask = da_t.where(mask == 0)
    #Select region around ASL
    da_mask = slice_region(da_mask, asl_region)
    #Search ASL and generate indices 
    all_lows_df  = get_lows(da_mask, asl_region, asl_min_dist, asl_num_peak, asl_exc_bord)
    all_lows_dfs = pd.concat([all_lows_dfs, all_lows_df], ignore_index=True)
    del(da_t,all_lows_df)
  asl_df = pd.DataFrame()
  asl_df = define_asl(times,all_lows_dfs, asl_region, l_allow_no_asl)
  
  #save data to text file
  asl_df.to_csv(os.path.join(out_path,'asl_scotthosking_index_v3_{}.csv'.format(case)), 
                index=False)

  #plot map with asl location mark
  ### slice area around ASL region
  da_nmsk = slice_region(da, asl_region)
  #Select region and apply land-sea mask
  da_mask = da.where(mask == 0)
  da_mask = slice_region(da_mask, asl_region)
  plot_asl_map(fig_path, da_nmsk, da_mask, asl_df, case)
  return

def plot_asl_map(fig_path, da, da_mask, asl_df, case):
  plt.figure(figsize=(20,15))
  for i in range(0,12):
    da_2D = da_mask.isel(time=i)
    da_2D = da_2D.sel(latitude=slice(-90,-55),
                      longitude=slice(165,305))
    ax = plt.subplot( 3, 4, i+1, 
                     projection = ccrs.Stereographic(
                                  central_longitude=0., 
                                  central_latitude=-90.) )
    ax.set_extent([165,305,-85,-55], ccrs.PlateCarree())
    result = da_2D.plot.contourf( 'longitude', 'latitude', cmap='Reds', 
                                  transform=ccrs.PlateCarree(), 
                                  add_colorbar=False, 
                                  levels=np.linspace(
                                         np.nanmin(da_2D.values), 
                                         np.nanmax(da_2D.values), 20) )
    # ax.coastlines(resolution='110m')
    ax.set_title(str(da_2D.time.values)[0:7])

    ## mark ASL
    df2 = asl_df[ asl_df['time'] == str(da_2D.time.values)[0:10]]
    if len(df2) > 0:
      ax.plot(df2['lon'], df2['lat'], 'mx', transform=ccrs.PlateCarree() )
    draw_regional_box(asl_region)
  
    plt.savefig(os.path.join(fig_path, "2d_map_asl_location_{}_{:02d}.png".format(case,i+1)))
    plt.close()

  return

# --- Function to read E3SM Diags metrics for CMIP6 models ---
def read_e3sm_diags_metrics(path, variables, seasons, names=None):
  # List of available models
  models = []
  paths = []
  dirs = sorted( glob.glob(path + os.path.sep) )
  for d in dirs:
      tmp = d.split(os.path.sep)
      model = tmp[-6]
      paths.append(d)
      models.append(model)
  if names:
      models = names

  # Array to hold data
  nmodels = len(models)
  nvariables = len(variables)
  nseasons = len(seasons)
  data = ma.array(np.zeros((nmodels,nvariables,nseasons)), mask=True)

  # Fill data
  for imodel in range(nmodels):
    for iseason in range(nseasons):
      # Open metrics file
      fname = paths[imodel]+'/%s_metrics_table.csv' % (seasons[iseason])
      with open(fname, 'r') as f:
          content = f.readlines()
          for ivariable in range(nvariables):
              # Skip of model has been flagged for this variable
              if models[imodel] in variables[ivariable]['exclude']:
                print("Excluding: %s, %s, %s" % (models[imodel],variables[ivariable]['name'],seasons[iseason]) )
                continue
              lines = [ l for l in content if l.startswith(variables[ivariable]['id']) ]
              if len(lines) > 1:
                raise "Found unexpected multiple entries"
              elif len(lines) == 1:
                rmse = lines[0].split(',')[-2]
                if rmse.upper() == 'NAN':
                  print("NAN: %s, %s, %s" % (models[imodel],variables[ivariable]['name'],seasons[iseason]) )
                else:
                  data[imodel,ivariable,iseason] = float(rmse)
                  #print(float(rmse),models[imodel])
              else:
                 print("Missing: %s, %s, %s" % (models[imodel],variables[ivariable]['name'],seasons[iseason]) )

  # Dictionary to hold data
  d = {}
  d['data'] = data.copy()
  d['models'] = models.copy()
  d['variables'] = variables.copy()
  d['seasons'] = seasons.copy()
  return d

def get_varible( ):
  # Variables
  variables = [
  {'name':'Net TOA',
   'units':'W m$^{-2}$',
   'id':'RESTOM global ceres_ebaf_toa_v4.1',
   'exclude':()},
  {'name':'SW CRE',
   'units':'W m$^{-2}$',
   'id':'SWCF global ceres_ebaf_toa_v4.1',
   'exclude':()},
  {'name':'LW CRE',
   'units':'W m$^{-2}$',
   'id':'LWCF global ceres_ebaf_toa_v4.1',
   'exclude':()},
  {'name':'prec',
   'units':'mm day$^{-1}$',
   'id':'PRECT global GPCP_v2.3',
   'exclude':('CIESM',)},
  {'name':'tas land',
   'units':'K',
   'id':'TREFHT land ERA5',
   'exclude':()},
  {'name':'SLP',
   'units':'hPa',
   'id':'PSL global ERA5',
   'exclude':()},
  {'name':'u-200',
   'units':'m s$^{-1}$',
   'id':'U-200mb global ERA5',
   'exclude':()},
  {'name':'u-850',
   'units':'m s$^{-1}$',
   'id':'U-850mb global ERA5',
   'exclude':()},
  {'name':'Zg-500',
   'units':'hm',
   'id':'Z3-500mb global ERA5',
   'exclude':('KIOST-ESM',)}, ]

  return variables 

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

  top_path  = "/lcrc/group/e3sm/ac.szhang/acme_scratch/e3sm_project/E3SMv21_testings/paper_material"
  run_path  = os.path.join(top_path,"fig_data","rmse_global")
  fig_path  = os.path.join(top_path,"rmse_bar")

  exps      = [ "cmip6", "v2_1.SORRM.control", "v2_1.SORRM.historical" ]
  labels    = [ "CMIP6", "CTRL-ENS",           "HIST-ENS" ]
  main(exps,labels,run_path,fig_path) 

