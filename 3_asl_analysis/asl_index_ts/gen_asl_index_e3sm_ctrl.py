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

def main(fig_path,out_path,mip,exp,relm,case_id,period,case_dict,asl_region,asl_min_dist, 
         asl_num_peak,asl_exc_bord,l_check_asl_region,l_allow_no_asl):
  for key in case_dict:
    if key == "mask":
      dmsk = case_dict[key].path
      vmsk = case_dict[key]._var
    else:
      case = key 
      var  = case_dict[key]._var
      data = case_dict[key].path
  
  print("working on ",case,var)

  # load model data 
  ds   = xr.open_dataset(data)
  #select target period 
  ymds = '{}-{}-01'.format(period.split("-")[0][0:4],period.split("-")[0][4:6])
  ymde = '{}-{}-31'.format(period.split("-")[1][0:4],period.split("-")[1][4:6])
  ds   = ds.sel(time=slice(ymds,ymde))
  if len(ds.dims) < 3: 
    print("data dimension is incorrect")
    exit()
  else:
    if ds[var].dims[1] == "lat" or ds[var].dims[2] == "lon":
      ds = ds.rename({ds[var].dims[1] : 'latitude',
                      ds[var].dims[2] : 'longitude'})

  #load land/sea mask file
  if os.path.exists(dmsk):
    dsm  = xr.open_dataset(dmsk)
    if dsm[vmsk].dims[0] == "lat" or dsm[vmsk].dims[1] == "lon":
      dsm = dsm.rename({dsm[vmsk].dims[0] : 'latitude',
                        dsm[vmsk].dims[1] : 'longitude'})
    mask = dsm[vmsk]
    mask = mask /100.0 #range[0,1]
    #print(mask.min(),mask.max())
  else:
    print("Warning: land/sea mask not exist, derive it...")
    #mask = cdutil.generateLandSeaMask(clt)
    lons, lats = ds.longitude, ds.latitude
    if lons.values.min() > -1:
      lons = ( (lons + 180) % 360 - 180)
    # Make a grid
    lon_grid, lat_grid = np.meshgrid(lons.values,lats.values)
    #global land mask
    mask = globe.is_land(lat_grid, lon_grid)
    del(lons,lats,lon_grid,lat_grid)

  #extract pressure data  
  da   = ds[var]
  if da.units == "Pa":
    # change units
    print("change units: from ", da.units, " to ", "hPa")
    da = da / 100. 
    da = da.assign_attrs(units='hPa')
  vstr = da.name.upper()
  vunt = da.units
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

  #Loop through each month and identify lows 
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
    all_lows_df  = get_lows(da_mask,asl_region,asl_min_dist,asl_num_peak,asl_exc_bord)
    all_lows_dfs = pd.concat([all_lows_dfs, all_lows_df], ignore_index=True)
    del(da_t,all_lows_df)

  asl_df = pd.DataFrame()
  asl_df,colnams = define_asl(times,all_lows_dfs,asl_region,l_allow_no_asl,mip,exp,relm,case,case_id,period,vstr,out_path)

  #plot map with asl location mark
  draw_asl_loc(da,mask,asl_df,asl_region,mip,exp,relm,case,case_id,period,vstr,vunt,fig_path)

  #plot asl time series 
  draw_asl_ts(fig_path,asl_df,colnams,asl_region,mip,exp,relm,case,case_id,period,vstr,vunt)

  #plot regression map 
  draw_asl_map(out_path,fig_path,ds[var],asl_df,colnams,asl_region,mip,exp,relm,case,case_id,period,vstr,vunt)

  return


def low_pass(cutoff_freq, data, order=5, axis=-1):
    #low-pass: filtering high-frequency signal
    #nyquist normalized cutoff for digital design
    Wn = cutoff_freq
    b, a = butter(order, Wn, btype='lowpass', analog=False)
    data_filt = filtfilt(b, a, data, axis=axis, method="gust")
    return data_filt

# -- Detrending
def detrend_dim(da, dim, deg=1):
    # detrend along a single dimension
    p = da.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    return da - fit

def draw_asl_ts(fig_path,asl_df,colnams,asl_region,mip,exp,relm,case,case_id,period,var,vunt):
  time  = asl_df['time']
  xtime = np.linspace(1,len(time),len(time))
  years = int(time[0].split("-")[0])
  yeare = int(time[len(time)-1].split("-")[0])

  # Plot all timeseries
  xtick = np.arange(0,len(time))
  xlabs = np.arange(0,len(time)) / 12.0  + years
  fontsize = 16
  fig = plt.figure(figsize=(8, 11))
  for i,col in enumerate(colnams[1:]):
    print("plot {}".format(col))
    var0 = np.array(asl_df[col])
    var1 = low_pass(1.0/11.0,var0,axis=0)
    ax = fig.add_subplot(len(colnams), 1, i+1)
    ax.plot(xtime, var0, color='grey', alpha=1.0, linewidth=0.8, label='monthly')
    ax.plot(xtime, var1, color='black', alpha=1.0, linewidth=1.4, label='11-point Hamming')
    # ax.set_title('Time series of monthly mean ASL longitude index (v3)')
    ax.set_xlabel('Time (years)')
    ax.set_ylabel(col)
    #ax.set_ylim(-4, 4)
    ax.set_xlim(0,len(time))
    ax.set_xticks(xtick[::120])
    ax.set_xticklabels(xlabs[::120].astype(int))
    ax.grid(True)
    if i+1 == len(colnams):
        ax.legend(loc='lower right', prop={'size':8})
    del(var0,var1)
  plt.suptitle('ASL indices ({},{})'.format(var,vunt), fontsize=fontsize*1.1)

  #plt.draw()
  plt.tight_layout()

  if not os.path.exists(fig_path):
    os.makedirs(fig_path)
  fig_name = "fig_ts_{}_{}_{}_{}_{}_{}_{}.pdf".format(mip,exp,case,relm,case_id,var,period)
  plt.savefig(os.path.join(fig_path, fig_name))
  plt.close()

  return

def draw_asl_map(out_path,fig_path,da,asl_df,colnams,asl_region,mip,exp,relm,case,case_id,period,var,vunt):
  lons  = da['longitude'][:]
  lats  = da['latitude'][:]
  clm   = da.groupby('time.month').mean(dim='time')
  anm   = (da.groupby('time.month') - clm)
  vstr  = da.name
  vunt  = da.units

  #nt,nlat,nlon = anm.shape
  #ngrd         = nlat*nlon
  #nyr          = nt/12
  #print(nt,nlat,nlon,ngrd,nyr)
  #ssta_series  = asl_df['asl_idx(degC)']  #.reshape(len(time))
  #rssta_series = asl_df['rasl_idx(degC)'] #.reshape(len(time))
  #asl_series  = asl_df['samSD_idx(1)']   #.reshape(len(time))
  #rasl_series = asl_df['raslSD_idx(1)']  #.reshape(len(time))
  time  = asl_df['time']
  xtime = np.linspace(1,len(time),len(time))
  asl_df = asl_df.set_index('time')
  asl_exist = False
  for col in colnams:
    if 'RelCenPres' in col:
      asl = asl_df[col].to_xarray()
      asl_exist = True
  if not asl_exist:
    exit('RelCenPres not exist, please check....')

  asl    = asl.assign_coords({"time": da.time})
  aslSD  = asl/asl.std(dim='time')

  years  = int(time[0].split("-")[0])
  yeare  = int(time[len(time)-1].split("-")[0])

  # -- filtering (or Running mean)
  raslSD    = aslSD
  ranm      = anm
  raslSD[:] = low_pass(1.0/5.0,aslSD,axis=0) #5-month: 1/5.0
  if np.isnan(anm).any():
    tmp1 = ranm.fillna(-99999)
    tmp1 = low_pass(1.0/5.0,tmp1,axis=0)
    ranm = anm
    ranm[:,:,:] = tmp1[:,:,:]
    ranm = ranm.where(ranm > -10000)
    del(tmp1)
  else:
    ranm[:,:,:] = low_pass(1.0/5.0,anm[:,:,:],axis=0)
  rdanm = detrend_dim(ranm,'time',1)
  #print(np.nanmax(ranm),np.nanmin(ranm))
  #print(np.nanmax(rdanm),np.nanmin(rdanm))

  # simultaneous
  #vreg,pval = pearsonr(raslSD,rdanm,axis=0)
  vcor = xr.corr(raslSD, rdanm, dim="time")
  vreg = xr.cov(raslSD, rdanm, dim="time")/raslSD.var(dim='time',skipna=True).values
  pval = xs.pearson_r_p_value(raslSD, rdanm, dim="time",skipna=True)

  #correlation at 5% significant level
  vsig = vreg.copy()
  #vsig = np.ma.masked_array(vreg, mask=(pval>=0.05)) #np.where(pval >= 0.05,np.nan,vreg)
  vsig = vsig.where(pval<=0.05)

  save_figure_data(vcor,vreg,pval,vsig,mip,exp,relm,case,case_id,period,var,vunt,out_path)

  # Show the plot
  vmin  = -1.0
  vmax  =  1.0
  nlev  =  21
  fontsize = 16
  bartitle = 'Regressed {}({})'.format(vstr,vunt)
  fig = plt.figure(figsize=(10, 12))
  grid = fig.add_gridspec(ncols=1, nrows=1)
  ax1, fill1 = draw_regression_map(vstr,vunt,lons,lats,vcor,vreg,pval,fig,asl_region,
                                   fontsize,'SAM-PSL Pattern',grid[0,0],vmin,vmax,nlev)
  cb = fig.colorbar(fill1,
                    ax=[ax1],
                    drawedges=True,
                    orientation='horizontal',
                    shrink=0.95,
                    aspect=40,
                    pad=0.05,
                    extendfrac='auto',
                    extendrect=True)
  ticks  = np.linspace(vmin, vmax, nlev)
  labels = []
  for i,tick in enumerate(ticks):
    if i % 2 == 0:
      labels.append('{:0.1f}'.format(tick))
    else:
      labels.append('')
  cb.set_ticks(ticks=ticks,labels=labels,fontsize=fontsize*0.9)
  cb.set_label(label=bartitle,fontsize=fontsize*0.95)

  #fig.suptitle('SST correlation & regression with Nino3.4', fontsize=fontsize, y=0.9)
  plt.rcParams["font.family"] = "sans-serif"
  plt.rcParams.update({'font.size': fontsize})

  plt.draw()

  if not os.path.exists(fig_path):
    os.makedirs(fig_path)
  fig_name = "fig_2d_map_{}_{}_{}_{}_{}_{}_{}.pdf".format(mip,exp,case,relm,case_id,var,period)
  plt.savefig(os.path.join(fig_path, fig_name))
  plt.close()

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
  #df2 = asl_df[ asl_df['time'] == str(da_2D.time.values)[0:10]]
  #if len(df2) > 0:
  #  ax.plot(df2['lon'], df2['lat'], 'mx', transform=ccrs.PlateCarree() )

  #add a box for sam region
  draw_regional_box(region)
  #draw_screen_poly(asl_region,m)

  #turn off ticks on top and right
  ax.xaxis.tick_bottom()
  ax.yaxis.tick_left()

  return ax, fillplot

def draw_asl_loc(da,mask,asl_df,asl_region,mip,exp,relm,case,case_id,period,vstr,vunt,fig_path):
  ### slice area around ASL region
  da_nmsk = slice_region(da, asl_region)
  #Select region and apply land-sea mask
  da_mask = da.where(mask == 0)
  da_mask = slice_region(da_mask, asl_region)

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
    ax.set_title('{}({}): {}'.format(var,vunt,str(da_2D.time.values)[0:7]))

    ## mark ASL
    df2 = asl_df[ asl_df['time'] == str(da_2D.time.values)[0:10]]
    if len(df2) > 0:
      ax.plot(df2['lon'], df2['lat'], 'mx', transform=ccrs.PlateCarree() )
    draw_regional_box(asl_region)
    
  if not os.path.exists(fig_path):
    os.makedirs(fig_path)
  fig_name = "fig_aslloc_map_{}_{}_{}_{}_{}_{}_1-12month.pdf".format(mip,exp,case,relm,case_id,var)
  plt.savefig(os.path.join(fig_path, fig_name))
  plt.close()

  del(da_nmsk,da_mask)

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
  plt.savefig(os.path.join(fig_path, "asl_region_{}.pdf".format(case)))
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

def define_asl(times,df,region,l_allow_no_asl,mip,exp,relm,case,case_id,period,vstr,out_path):
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
  colnams = ['time','lon','lat','ActCenPres','SectorPres','RelCenPres']
  df2 = df2.sort_values(by="time")
  df2 = df2.reset_index(drop=True)
  df2 = df2[colnams]

  #save data to text file
  if not os.path.exists(out_path):
    os.makedirs(out_path)
  out_file = '{}.{}.{}.{}.{}.{}.{}.csv'.format(mip,exp,case,relm,case_id,vstr,period)
  df2.to_csv(os.path.join(out_path,out_file),index=False)
  return df2,colnams

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
  out_path = os.path.join(top_path,"E3SMv21_testings","paper_material","fig_data","asl_analysis","raw_index")
  fig_path = os.path.join(top_path,"E3SMv21_testings","paper_material","3_asl_analysis","asl_index_ts","figure")

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

  mip       = "e3sm"
  exps      = [ "piControl"]
  relms     = [ "1950"]
  products  = ["v2_1-SORRM"]
  tableId   = "Amon"
  var       = "psl"
  varstr    = "PSL"
  case_id   = 'asl_scotthoskingv3'
  period    = "080101-100012"
  run_path  = "/lcrc/group/e3sm/ac.szhang/acme_scratch/data/pcmdi/model/monthly"
  run_mask  = "/lcrc/group/e3sm/ac.szhang/acme_scratch/data/pcmdi/model/fixed/sftlf"

  for exp in exps:
    for relm in relms:
      for product in products:
        case = product
        ptmp = os.path.join(run_path,mip,exp,tableId,var)
        ftmp = '{}.{}.{}.{}.*.{}.{}.nc'.format(mip,exp,product,relm,var,period)
        filePath = sorted(glob.glob(os.path.join(ptmp,ftmp)))
        if len(filePath) > 0 and os.path.isfile(filePath[0]):
          print("case: ", case)
          print("data: ", filePath[0])
          fileName = filePath[0].split("/")[-1]
          asl_min_dist = 1  # peaks are separated by at least min_distance
          asl_num_peak = 12 # maximum number of peaks
          data_fil = filePath[0]
          mask_fil = os.path.join(run_mask,tableId,mip+"."+exp+"."+product+".fx.sftlf.nc")

          case_dict = collections.OrderedDict()
          case_dict[case] = Case(data_fil, var= var.upper(), color="blue", label=case)
          case_dict['mask'] = Case(mask_fil, var= "sftlf", color="blue", label=case)

          #call fuction to generate ASL index 
          main(fig_path,out_path,mip,exp,relm,case_id,period,case_dict,asl_region,asl_min_dist, 
               asl_num_peak,asl_exc_bord,l_check_asl_region,l_allow_no_asl)
