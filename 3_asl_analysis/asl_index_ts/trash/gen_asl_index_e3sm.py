import os
import collections
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

def main(fig_path, out_path, case_dict, asl_region, asl_min_dist, 
         asl_num_peak, asl_exc_bord,l_check_asl_region, l_allow_no_asl):
  for key in case_dict:
    if key == "mask":
      dmsk = case_dict[key].path
      vmsk = case_dict[key]._var
    else:
      case = key 
      var  = case_dict[key]._var
      data = case_dict[key].path
  
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

def draw_regional_box( region, transform=None ):
  '''
  Draw box around a region on a map
  region is a dictionary with west,east,south,north
  '''

  if transform == None:
      transform = ccrs.PlateCarree()

  plt.plot([region['west'], region['west']], [region['south'],region['north']], 
               'k-', transform=transform, linewidth=1)
  plt.plot([region['east'], region['east']], [region['south'],region['north']], 
               'k-', transform=transform, linewidth=1)
  
  for i in range( np.int32(region['west']),np.int32(region['east']) ): 
      plt.plot([i,i+1], [region['south'],region['south']], 'k-', transform=transform, linewidth=1)
      plt.plot([i,i+1], [region['north'],region['north']], 'k-', transform=transform, linewidth=1)

def plot_regions_mask(fig_path, da, da_mask, group):
  if not os.path.exists(fig_path):
      os.makedirs(fig_path)
  plt.figure(figsize=(5,5))
  ax1 = plt.subplot(121, projection=ccrs.Stereographic(central_longitude=0., central_latitude=-90.) )
  ax1.set_extent([-180,180,-90,-50], ccrs.PlateCarree())
  da.isel(time=-1).plot.pcolormesh('longitude', 'latitude', cmap='jet', 
                                   transform=ccrs.PlateCarree(), add_colorbar=False )
  ax2 = plt.subplot(122, projection=ccrs.Stereographic(central_longitude=0., central_latitude=-90.) )
  ax2.set_extent([-180,180,-90,-50], ccrs.PlateCarree())
  da_mask.isel(time=-1).plot.pcolormesh('longitude', 'latitude', cmap='jet', 
                                        transform=ccrs.PlateCarree(), add_colorbar=False )
  plt.savefig(os.path.join(fig_path, "asl_region_{}.png".format(case)))
  plt.close()
  return

def asl_sector_mean(da, region):
  latn = region['north'] 
  lats = region['south'] 
  lone = region['east']  
  lonw = region['west']  
  #print("average over region for ASL: latn,lats,lone,lonw=",latn,lats,lonw,lone)
  a = da.sel(latitude=slice(lats,latn),
             longitude=slice(lonw,lone)).mean().values
  return a

def get_lows(da, asl_region, min_dist, num_peak, exclue_border):
  '''
  da for one point in time (with lats x lons)
  '''
  import scipy.ndimage as ndimage

  lons, lats = da.longitude.values, da.latitude.values
  sector_mean_pres = asl_sector_mean(da, asl_region)
  threshold = sector_mean_pres
  time_str = str(da.time.values)[:10]
  # fill land in with highest value to limit lows being found here
  da_max = da.max().values
  da = da.fillna(da_max)

  invert_data = (da*-1.).values     # search for peaks rather than minima

  #apply a filtering to smooth field 
  #img = ndimage.gaussian_filter(da.values, sigma=5)
  #img = ndimage.maximum_filter(invert_data, size=min_dist*0.5, mode='constant')
  #invert_data = img

  if threshold is None:
    threshold_abs = invert_data.mean()
  else:
    threshold_abs  = threshold * -1  # define threshold cut-off for peaks (inverted lows)
  minima_yx = peak_local_max(invert_data,              # input data
                         min_distance=min_dist,        # peaks are separated by at least min_distance
                         num_peaks=num_peak,           # maximum number of peaks
                         exclude_border=exclue_border, # excludes peaks from within min_distance pixels of the border
                         threshold_abs=threshold_abs   # minimum intensity of peaks
                         )
  minima_lat, minima_lon, pressure = [], [], []
  for minima in minima_yx:
      minima_lat.append(lats[minima[0]])
      minima_lon.append(lons[minima[1]])
      pressure.append(da.values[minima[0],minima[1]])
  
  df = pd.DataFrame()
  df['lat']        = minima_lat
  df['lon']        = minima_lon
  df['ActCenPres'] = pressure
  df['SectorPres'] = sector_mean_pres
  df['time']       = time_str
  
  ### Add relative central pressure (Hosking et al. 2013)
  df['RelCenPres'] = df['ActCenPres'] - df['SectorPres']
  
  ### re-order columns
  df = df[['time','lon','lat','ActCenPres','SectorPres','RelCenPres']]
  
  ### clean-up DataFrame
  df = df.reset_index(drop=True)

  return df

def define_asl(times, df, region, l_allow_no_asl):
  ### select only those points within ASL box
  df2 = df[(df['lon'] > region['west'])  & 
           (df['lon'] < region['east'])  & 
           (df['lat'] > region['south']) & 
           (df['lat'] < region['north']) ]
  ### For each time, get the row with the lowest minima_number
  df2 = df2.loc[df2.groupby('time')['ActCenPres'].idxmin()]
  df2 = df2.reset_index(drop=True)
  if len(df2) != len(times):
    print("WARNNING from define_asl()")
    print("length of time in data: ", len(times))
    print("length of time with identified ASL: ", len(df2)) 
    print("there are missing time steps in asl data, can not continue...")
    c =list(set(df2['time']).symmetric_difference(times.values))
    print("no asl present in following months:",c)
    if l_allow_no_asl:
      for time_str in c:  
        #option 1: fill missing values to non-asl months
        asl_df               = pd.DataFrame()
        asl_df['lat']        = [np.nan]
        asl_df['lon']        = [np.nan]
        asl_df['ActCenPres'] = [np.nan]
        asl_df['SectorPres'] = [np.nan]
        asl_df['RelCenPres'] = [np.nan]
        asl_df['time']       = time_str
        df2 = pd.concat([df2, asl_df], ignore_index=True)
        del(asl_df)
    else:
      ### find the lowest minima_number in the raw data
      df1 = df.loc[df.groupby('time')['ActCenPres'].idxmin()]
      df1 = df1.reset_index(drop=True)
      for time_str in c:
        indx = list(df1['time']).index(time_str)
        asl_df               = pd.DataFrame()
        asl_df['lat']        = [df1['lat'][indx]] 
        asl_df['lon']        = [df1['lon'][indx]]
        asl_df['ActCenPres'] = [df1['ActCenPres'][indx]]
        asl_df['SectorPres'] = [df1['SectorPres'][indx]]
        asl_df['RelCenPres'] = [df1['RelCenPres'][indx]]
        asl_df['time']       = time_str
        df2 = pd.concat([df2, asl_df], sort=True) #ignore_index=True,sort=True)
        del(asl_df,indx)
      del(df1)
  
  ### re-order columns
  df2 = df2.sort_values(by="time")
  df2 = df2.reset_index(drop=True)
  df2 = df2[['time','lon','lat','ActCenPres','SectorPres','RelCenPres']]
  return df2

def slice_region(da, region, boarder=8):
  latn = region['north'] + boarder
  lats = region['south'] - boarder
  lone = region['east']  + boarder
  lonw = region['west']  - boarder
  #print("select region for ASL: latn,lats,lone,lonw=",latn,lats,lonw,lone)

  da = da.sel(latitude=slice(lats,latn),longitude=slice(lonw,lone))
  return da

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
  out_path = os.path.join(top_path,"E3SMv21_testings","paper_material","fig_data","asl_analysis")
  fig_path = os.path.join(out_path,"figure")

  # region of interest (asl sector)
  asl_region   = {'west':170., 'east':298., 'south':-80., 'north':-60.}
  # tunable parameters for ASL search. They need to be adjusted with the model
  # resolution and if you want to have non-missing ASLs. 
  asl_min_dist = 5 # peaks are separated by at least min_distance
  asl_num_peak = 3 # maximum number of peaks
  asl_exc_bord = False # excludes peaks from within min_distance pixels of the border

  #sanity check (plot region with and without mask) 
  l_check_asl_region = False #True  
  #if asl is not identified in the selected region, then search outside region 
  #and find a loction with local minimum pressure when l_allow_no_asl = False.
  # if l_allow_no_asl = True, then missing values will be asigned if asl is not found  
  l_allow_no_asl = False #True 

  #cases = [ "v2_1.SORRM.histssp370_0701",
  #          "v2_1.SORRM.histssp370_0751", 
  #          "v2_1.SORRM.histssp370_0801"]
  #labels = [ "0701", "0751", "0801"]
  #run_path = "/lcrc/group/e3sm/ac.szhang/acme_scratch/e3sm_project/E3SMv21_testings" 
  #mask = "/lcrc/group/e3sm/ac.szhang/acme_scratch/data/pcmdi/model/fixed/sftlf/Amon/e3sm.historical.v2_1-SORRM.fx.sftlf.nc"

  cases     = ["v2_1.SORRM.control","v2_1.SORRM.histssp370_0701","v2_1.SORRM.histssp370_0751",
               "v2_1.SORRM.histssp370_0801","v2_1.SORRM.histssp370-fismf_0701"]
  labels    = ["CTRL","HIST-701","HIST-751","HIST-801","HIST-fismf"]
  run_path  = "/lcrc/group/e3sm/ac.szhang/acme_scratch/e3sm_project/E3SMv21_testings"
  run_mask  = "/lcrc/group/e3sm/ac.szhang/acme_scratch/data/pcmdi/observations/fixed/sftlf"
  case_dict = collections.OrderedDict()

  for i,case in enumerate(cases):
    asl_min_dist = 1  # peaks are separated by at least min_distance
    asl_num_peak = 24 # maximum number of peaks
    if labels[i] == "CTRL": 
      data_fil = os.path.join(run_path,case,"post/atm/180x360_aave/ts/monthly/50yr/PSL_*")
    else: 
      data_fil = os.path.join(run_path,case,"post/atm/180x360_aave/ts/monthly/10yr/PSL_*")
    print(data_fil)
    exit()
    mask_fil = os.path.join(run_mask,case+".fx.sftlf.nc")
    case_dict[case] = Case(data_fil, var= "PSL", color="blue", label=labels[i])
    case_dict['mask'] = Case(mask_fil, var= "sftlf", color="blue", label=labels[i])

    #call fuction to generate ASL index 
    main(fig_path, out_path, case_dict, asl_region, asl_min_dist, 
         asl_num_peak, asl_exc_bord, l_check_asl_region, l_allow_no_asl)

