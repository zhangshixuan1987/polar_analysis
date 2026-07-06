import os
import glob 
import collections
import xarray as xr
import xskillscore as xs
import numpy as np
import pandas as pd
from datetime import datetime
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from scipy.stats import pearsonr
from scipy.signal import butter, filtfilt, sosfilt,lfilter
from mpl_toolkits.basemap import Basemap
from matplotlib.pylab import rcParams
from matplotlib.patches import Polygon
from global_land_mask import globe

import cmaps as gvcmaps
import geocat.viz.util as gvutil
import geocat.viz as gv

def main(fig_path,out_path,mip,exp,relm,case_id,period,case_dict,sam_region,l_check_sam_region):
  
  case = case_dict.keys()
  case = list(case)[0]
  var  = case_dict[case]._var
  data = case_dict[case].path
  
  print('working on {} {}'.format(case,var))

  #load data file 
  ds   = xr.open_dataset(data)
  #select target period 
  ymds = '{}-{}-01'.format(period.split("-")[0][0:4],period.split("-")[0][4:6])
  ymde = '{}-{}-31'.format(period.split("-")[1][0:4],period.split("-")[1][4:6])
  da   = ds.sel(time=slice(ymds,ymde))

  var  = 'psl'
  vstr = 'PSL'
  vunt = 'hPa'

  #calculate and save sam index 
  sam_df,colnams = define_sam(mip,exp,relm,case,case_id,period,vstr,vunt,da,sam_region,out_path)

  #plot sam time series 
  draw_sam_ts(fig_path,sam_df,colnams,sam_region,mip,exp,relm,case,case_id,period,vstr,vunt) 

  return

# -- Detrending
def detrend_dim(da, dim, deg=1):
    # detrend along a single dimension
    p = da.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    return da - fit

def draw_sam_ts(fig_path,sam_df,colnams,sam_region,mip,exp,relm,case,case_id,period,var,vunt): 

  time  = sam_df['time']
  xtime = np.linspace(1,len(time),len(time))
  for col in colnams:
    print(col)
    if 'sam_idx' in col: 
      ssta_series  = np.array(sam_df[col])  #.reshape(len(time))
    if 'rsam_idx' in col: 
      rssta_series = np.array(sam_df[col]) #.reshape(len(time))
    if 'samSD_idx' in col: 
      sam_series = np.array(sam_df[col]) #.reshape(len(time))
    if 'rsamSD_idx' in col: 
      rsam_series = np.array(sam_df[col]) #.reshape(len(time))
  years = int(time[0].split("-")[0])
  yeare = int(time[len(time)-1].split("-")[0])

  # -- figure plot
  fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))

  xtick = np.arange(0,len(time))
  xlabs = np.arange(0,len(time)) / 12.0  + years
  fontsize = 16

  ax1.plot(xtime, rssta_series, 'black', alpha=1.00, linewidth=2)
  ax1.fill_between(xtime, 0., ssta_series, ssta_series> 0., color='red',   alpha=.75)
  ax1.fill_between(xtime, 0., ssta_series, ssta_series< 0., color='blue',  alpha=.75)
  #ax1.xaxis.tick_top()
  ax1.set_title("Raw sam index {}({})".format(var,vunt), fontsize=fontsize*1.1)
  ax1.set_xlabel("years", fontsize=fontsize*1.1)
  ax1.set_ylabel("degC", fontsize=fontsize*1.1)
  ax1.legend(['5-month running mean'])
  ax1.set_ylim(-4, 4)
  ax1.set_xlim(0,len(time))
  ax1.set_xticks(xtick[::120])
  ax1.set_xticklabels(xlabs[::120].astype(int))
  ax1.tick_params(labelsize=fontsize)
  ax1.grid(True)

  ax2.plot(xtime, rsam_series, 'black', alpha=1.00, linewidth=2)
  ax2.fill_between(xtime, 0., sam_series, sam_series> 0., color='red',   alpha=.75)
  ax2.fill_between(xtime, 0., sam_series, sam_series< 0., color='blue',  alpha=.75)
  #ax2.xaxis.tick_top()
  ax2.set_title("Standardized sam index {}({})".format(var,vunt), fontsize=fontsize*1.1)
  ax2.set_xlabel("years", fontsize=fontsize*1.1)
  ax2.set_ylabel("Unitless", fontsize=fontsize*1.1)
  ax2.legend(['5-month running mean'])
  ax2.set_ylim(-4, 4)
  ax2.set_xlim(0,len(time))
  ax2.set_xticks(xtick[::120])
  ax2.set_xticklabels(xlabs[::120].astype(int))
  ax2.tick_params(labelsize=fontsize)
  ax2.grid(True)

  #plt.draw()
  plt.tight_layout()

  if not os.path.exists(fig_path):
    os.makedirs(fig_path)
  fig_name = "fig_ts_{}_{}_{}_{}_{}_{}_{}.pdf".format(mip,exp,case,relm,case_id,var,period)
  plt.savefig(os.path.join(fig_path, fig_name))
  plt.close()

  return

def save_figure_data(vcor,vreg,pval,vsig,mip,exp,relm,case,case_id,period,var,vunt,out_path):
  #create xarray to save data 
  lons,lats = vcor.longitude,vcor.latitude
  df = xr.Dataset(
          {   "var_reg": (["lat", "lon"], vreg.data),
              "var_sig": (["lat", "lon"], vsig.data),
              "var_cor": (["lat", "lon"], vcor.data),
              "pval": (["lat", "lon"], pval.data),
              },
          coords={
              "lon": (["lon"], lons.data),
              "lat": (["lat"], lats.data),
              },
          attrs=dict(description="sam regression maps",
                    reference_time=period), 
          )

  df.lat.attrs["units"] = "degree_north"
  df.lon.attrs["units"] = "degree_east"
  df.var_cor.attrs["units"] = "1"
  df.var_cor.attrs["long_name"] = var + " correlation"
  df.var_reg.attrs["units"] = vunt 
  df.var_reg.attrs["long_name"] = var + " regression"
  df.var_sig.attrs["units"] = vunt 
  df.var_sig.attrs["long_name"] = var + " regression (significant at 0.05 confidence level)"
  df.pval.attrs["units"] = "1"
  df.pval.attrs["long_name"] = "p values"
 
  #save data to netcdf 
  if not os.path.exists(out_path):
    os.makedirs(out_path)
  fout_name = "{}_{}_{}_{}_{}_{}_{}.nc".format(mip,exp,case,relm,case_id,var,period)
  df.to_netcdf(os.path.join(out_path,fout_name))

  return 

def draw_regression_map(vstr,vunt,lats,lons,cor,reg,pval,fig,region,fontsize,title,grid_space,vmin,vmax,nlev):
  sig    = pval
  sig[:] = 1.0 - sig[:]
  t90    = 0.94
  t95    = 0.95
  rlabel = '{}({})'.format(vstr,vunt)

  # Generate axes using Cartopy to draw coastlines
  ax = fig.add_subplot(grid_space,
          projection=ccrs.PlateCarree(central_longitude=210))
  ax.coastlines(linewidth=0.5, alpha=0.6)

  # Use geocat.viz.util convenience function to set axes limits & tick values
  gvutil.set_axes_limits_and_ticks(ax,
                                 xlim=(-180, 180),
                                 ylim=(-90, 90),
                                 xticks=np.arange(-180, 181, 60),
                                 yticks=np.arange(-90, 91, 30))

  # Use geocat.viz.util convenience function to add minor and major tick lines
  gvutil.add_major_minor_ticks(ax, labelsize=fontsize*0.90)

  # Use geocat.viz.util convenience function to make latitude, longitude tick labels
  gvutil.add_lat_lon_ticklabels(ax)

  # Import the default color map
  newcmp = gvcmaps.BlueYellowRed
  index = [5, 20,  35, 50, 65, 85, 95, 110, 125,  0, 0, 135, 150,  165, 180, 200, 210, 220, 235, 250 ]
  color_list = [newcmp[i].colors for i in index]
  #-- Change to white
  color_list[9]=[ 1., 1., 1.]
  color_list[10]=[ 1., 1., 1.]

  # Define dictionary for kwargs
  kwargs = dict(
    vmin = vmin,
    vmax = vmax,
    levels = nlev,
    colors=color_list,
    add_colorbar=False,  # allow for colorbar specification later
    transform=ccrs.PlateCarree(),  # ds projection
  )

  # Contouf-plot U data (for filled contours)
  fillplot = cor.plot.contourf(ax=ax,  **kwargs)

  # Draw map features on top of filled contour
  ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=1)
  ax.add_feature(cfeature.COASTLINE, edgecolor='gray', linewidth=0.5, zorder=1)

  # Plot Hatch for significance
  sig.plot.contourf(ax=ax, levels = [-1*t95, -1*t90, t90, t95], colors='none',
      hatches=[None, None, None, '..', '..'], extend='both',
      add_colorbar=False, transform=ccrs.PlateCarree())

  # Plot line contours
  # Specify contour levels excluding 0
  delc=0.2
  levels = np.arange(-3, 0, delc)
  levels = np.append(levels, np.arange(delc, 3, delc))

  rad = reg.plot.contour(ax=ax,
                  colors='black',
                  alpha=0.8,
                  linewidths=1.0,
                  add_labels=False,
                  levels=levels,
                  transform=ccrs.PlateCarree())
#  ax.clabel(rad, levels, fmt='%1.1f',  inline=True, colors='black', fontsize=fontsize)
  pe = [PathEffects.withStroke(linewidth=2.0, foreground="w")]
  plt.setp(rad.collections, path_effects=pe)

  # Use geocat.viz.util convenience function to add titles to left and right of the plot axis.
  gvutil.set_titles_and_labels(ax,
                             lefttitle=title,
                             lefttitlefontsize=fontsize*0.95,
                             righttitle=rlabel,
                             righttitlefontsize=fontsize*0.95,
                             labelfontsize = fontsize*0.95,
                             xlabel="",
                             ylabel="")

  # add mark for specific location 
  #df2 = sam_df[ sam_df['time'] == str(da_2D.time.values)[0:10]]
  #if len(df2) > 0:
  #  ax.plot(df2['lon'], df2['lat'], 'mx', transform=ccrs.PlateCarree() )
  
  #add a box for sam region 
  draw_regional_box(region)
  #draw_screen_poly(sam_region,m)

  #turn off ticks on top and right
  ax.xaxis.tick_bottom()
  ax.yaxis.tick_left()

  return ax, fillplot

def draw_screen_poly(region, m):
  '''
  Draw box around a region on a map
  region is a dictionary with west,east,south,north
  '''
  lat0 = region['north']
  lat1 = region['south']
  lon0 = region['east']
  lon1 = region['west']
  if lon1 < 0.0: 
     lon1 + 360.0
  if lon0 < 0.0:
     lon0 + 360.0

  lats = [ lat1, lat0, lat0, lat1 ]
  lons = [ lon1, lon1, lon0, lon0 ]

  x, y = m(lons,lats)
  xy   = zip(x,y)

  poly = Polygon( xy, facecolor='red', alpha=0.4 )
  plt.gca().add_patch(poly)

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
  plt.figure(figsize=(12,6))
  ax1 = plt.subplot(211, projection=ccrs.PlateCarree(central_longitude=0.,globe=None) )
  ax1.set_extent([-180,110,-10,10], ccrs.PlateCarree())
  da.isel(time=-1).plot.pcolormesh('longitude', 'latitude', cmap='jet', 
                                   transform=ccrs.PlateCarree(), add_colorbar=False )
  ax2 = plt.subplot(212, projection=ccrs.PlateCarree(central_longitude=0.,globe=None) )
  ax2.set_extent([-180,110,-10,10], ccrs.PlateCarree())
  da_mask.isel(time=-1).plot.pcolormesh('longitude', 'latitude', cmap='jet', 
                                        transform=ccrs.PlateCarree(), add_colorbar=False )
  plt.savefig(os.path.join(fig_path, "sam_region_{}.pdf".format(case)))
  plt.close()
  return

def low_pass(cutoff_freq, data, order=5, axis=-1):
    #low-pass: filtering high-frequency signal
    #nyquist normalized cutoff for digital design
    Wn = cutoff_freq
    b, a = butter(order, Wn, btype='lowpass', analog=False)
    data_filt = filtfilt(b, a, data, axis=axis, method="gust")
    return data_filt

def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295
    hav = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p)*cos(lat2*p) * (1-cos((lon2-lon1)*p)) / 2
    return 12742 * asin(sqrt(hav))

def closest_xy(data, v):
    return min(data, key=lambda p: distance(v['lat'],v['lon'],p['lat'],p['lon']))

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx],idx

def define_sam(mip,exp,relm,case,case_id,period,vstr,vunt,da,region,out_path):
  '''
  da for one point in time (with lats x lons)
  Based on the numerical definition of the SAM by Gong and Wang (1999), which is:
  SAM = P*40°S – P*65°S
  where P*40°S and P*65°S are the normalized monthly zonal sea level pressure (SLP) at 40°S and 65°S, respectively, 
  see Marshall (2003) and at the dataset website: https://legacy.bas.ac.uk/met/gjma/sam.html. 
  '''
  #years = list(da.time.dt.year.data)
  times = da.time.dt.strftime("%Y-%m-%d")
  time_str = da.time.dt.strftime("%Y-%m-%d")
  ntime = len(times)
  print("number of total months in data: ", ntime)

  # extract index 
  sam_evar    = da['pc'].values.copy()
  sam_evar[:] = da['frac'].values * 100.0
  sam_ind     = da['pc'].values.copy()
  samSD_ind   = sam_ind / sam_ind.std()
  #instead of moving average, using a low-pass filtering
  #rsam_ind   = sam_ind.rolling(time=5, center=True).mean('time')
  #osam_ind   = sam_ind.rolling(time=3, center=True).mean('time')
  #rsamSD_ind = samSD_ind.rolling(time=5, center=True).mean('time')
  #osamSD_ind = samSD_ind.rolling(time=3, center=True).mean('time')
  rsam_ind    = low_pass(1.0/5.0,sam_ind,axis=0)   #5-month: 1/5.0
  osam_ind    = low_pass(1.0/3.0,sam_ind,axis=0)   #3-month: 1/3.0
  rsamSD_ind  = low_pass(1.0/5.0,samSD_ind,axis=0) #5-month: 1/5.0
  osamSD_ind  = low_pass(1.0/3.0,samSD_ind,axis=0) #3-month: 1/3.0

  colnams = ['time','sam_evar(%)','sam_idx(1)','rsam_idx(1)','osam_idx(1)','samSD_idx(1)','rsamSD_idx(1)','osamSD_idx(1)']
  df = pd.DataFrame([],columns=colnams)
  for col in colnams: 
    if 'time' in col: 
      df[col] = time_str
    elif 'sam_evar' in col:
      df[col] = sam_evar 
    elif 'sam_idx' in col:
      df[col] = sam_ind
    elif 'samSD_idx' in col:
      df[col] = samSD_ind
    elif 'rsam_idx' in col:
      df[col] = rsam_ind
    elif 'rsamSD_idx' in col:
      df[col] = rsamSD_ind
    elif 'osam_idx' in col:
      df[col] = osam_ind
    elif 'osamSD_idx' in col:
      df[col] = osamSD_ind
  ### clean-up DataFrame
  df = df.reset_index(drop=True)
  #save data to text file
  if not os.path.exists(out_path):
    os.makedirs(out_path)
  out_file = '{}.{}.{}.{}.{}.{}.{}.csv'.format(mip,exp,case,relm,case_id,vstr,period)
  df.to_csv(os.path.join(out_path,out_file),index=False)

  del(da,sam_evar,sam_ind,rsam_ind,samSD_ind,rsamSD_ind,times,time_str)

  return df,colnams 

def slice_region(da, region, boarder=1):
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
  data_dir = "/lcrc/group/e3sm/public_html/diagnostic_output/ac.szhang/E3SMv2_1/pcmdi"
  out_path = os.path.join("/lcrc/group/e3sm/public_html/diagnostic_output/ac.szhang/polar_analysis/data","sam_analysis","raw_index")
  fig_path = os.path.join("/lcrc/group/e3sm/public_html/diagnostic_output/ac.szhang/polar_analysis/figure", "sam_analysis","sam_index_ts","figure")

  # region of interest (sam sector)
  sam_region   = {'west':-180., 'east':180., 'south':-90., 'north':-20.}

  #sanity check (plot region with and without mask) 
  l_check_sam_region = False #True  

  mip       = "e3sm"
  exps      = [ "historical"]
  relms     = [ "0701", "0751", "0801" ]
  products  = ["v2_1-SORRM"]
  tableId   = "Amon"
  var       = "psl"
  varstr    = "PSL"
  case_id   = 'sam_cbf'
  period    = "195001-201412"
  run_path  = os.path.join(data_dir,'diagnostic_results/variability_modes')
  for exp in exps:
    for relm in relms:
      for product in products:
        case = product
        ptmp = os.path.join(run_path,mip,exp,'v20241111/SAM/NOAA-20C')
        ftmp = '{}_{}_EOF1_monthly_{}_{}_{}_{}_*_{}_{}_cbf.nc'.format(
                case_id.split("_")[0].upper(),varstr,mip,"*",exp,relm,relm,"*")
        print(ptmp+"/"+ftmp)
        filePath = sorted(glob.glob(os.path.join(ptmp,ftmp)))
        if len(filePath) > 0 and os.path.isfile(filePath[0]):
          fileName = filePath[0].split("/")[-1]
          data_fil = filePath[0]
          case_dict = collections.OrderedDict()
          case_dict[case] = Case(data_fil, var= var.upper(), color="blue", label=case)
          #call fuction to generate rgmn index 
          main(fig_path,out_path,mip,exp,relm,case_id,period,case_dict,sam_region,l_check_sam_region)
