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

from scipy import stats
from scipy.stats import pearsonr
from scipy.signal import butter, filtfilt, sosfilt,lfilter
from mpl_toolkits.basemap import Basemap
from matplotlib.pylab import rcParams
from matplotlib.patches import Polygon
from global_land_mask import globe

import cmaps as gvcmaps
import geocat.viz.util as gvutil
import geocat.viz as gv
import cmaps as gvcmaps
from geocat.comp import trendunc_trends, trendunc_pcs

def main(fig_path,out_path,mip,exp,relm,case_id,period,pclimo,case_dict,trend_region,regnam,season):  
  for key in case_dict:
    if key == "mask":
      dmsk = case_dict[key].path
      vmsk = case_dict[key]._var
    elif key == "clim": 
      dclim = case_dict[key].path
      vclim = case_dict[key]._var
    else:
      case = key 
      var  = case_dict[key]._var
      data = case_dict[key].path

  print("working on ",case,var)

  #load data file 
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

  #extract surface temperature data and mask out land region
  if (var == 'TS' or var == "ts" ):
    da = ds[var].where(mask == 0)
  else:
    da = ds[var]

  if (var == "TS" or var == "TREFHT"):
    if da.units == "K":
      # change units
      print("change units: from ", da.units, " to ", "degC")
      da = da - 273.15
    da = da.assign_attrs(units='$^{o}$C')
  elif var == "PRECT":
    if ( da.units == "m/s" or da.units == "m s~S~-1~N~" or da.units == "m s^{-1}"):
      # change units
      print("change units: from ", da.units, " to ", "mm/day")
      da = da * 1000.0 * 86400.0
    da = da.assign_attrs(units='mm day$^{-1}$')
  elif var == "PSL":
    if da.units == "Pa":
      # change units
      print("change units: from ", da.units, " to ", "hPa")
      da = da / 100.0
    da = da.assign_attrs(units='hPa')
  elif (var == "TAUX" or var == "TAUY"):
    if ( da.units == "N/m2" or da.units == "N m~S~-2~N~" or da.units == "N m^{-2}"):
      # change units
      print("change units: from ", da.units, " to ", "10^{-2}xPa")
      da = da * 100.0
    da = da.assign_attrs(units='10$^{-2}$xPa')

  #import scipy.stats as stats
  sst_grd  = sst.reshape((nt, ngrd), order='F')
  x        = np.linspace(1,nt,nt)#.reshape((nt,1))
  sst_rate = np.empty((ngrd,1))
  sst_rate[:,:] = np.nan
  
  for i in range(ngrd):
    y = sst_grd[:,i]
    if(not np.ma.is_masked(y)):
        z = np.polyfit(x, y, 1)
        sst_rate[i,0] = z[0]*120.0
        #slope, intercept, r_value, p_value, std_err = stats.linregress(x, sst_grd[:,i])
        #sst_rate[i,0] = slope*120.0
        
  sst_rate = sst_rate.reshape((nlat,nlon), order='F')

  #derive data for trend analysis
  if os.path.exists(dclim):
    l_prescribe_clim = True 
  else:
    l_prescribe_clim = False
  df = find_regional_data(da,trend_region,dclim,vclim,l_prescribe_clim,period,pclimo,season,mip,exp,relm,case,case_id,out_path)
  df.attrs['name'] = da.name
  df.attrs['units'] = da.units

  #calculate and save trend index 
  pcs,dout = trend_analysis(df,mip,exp,relm,case,case_id,period,out_path,fig_path)

  #plot trend time series 
  draw_trend_ts(fig_path,dout,pcs,trend_region,mip,exp,relm,case,case_id,period)

  #plot map with trend regression on TS
  draw_trend_map(out_path,fig_path,dout,pcs,trend_region,mip,exp,relm,case,case_id,period,da.name.upper(),da.units)
  
  return

# -- Detrending
def detrend_dim(da, dim, deg=1):
    # detrend along a single dimension
    p = da.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    return da - fit

def draw_trend_ts(fig_path,da,trend_df,trend_region,mip,exp,relm,case,case_id,period):
  
  var = da.name #coords['name']
  vstr = da.name.upper() 
  vunt = da.units #.values

  time  = trend_df['time']
  xtime = np.linspace(1,len(time),len(time))
  years = int(time[0].split("-")[0])
  yeare = int(time[len(time)-1].split("-")[0])

  # -- figure plot
  fig, ax1 = plt.subplots(5, 1, figsize=(12, 6))

  xtick = np.arange(0,len(time))
  xlabs = np.arange(0,len(time)) / 12.0  + years
  fontsize = 16

  for itrend in range(0,5):
    ssta_series = trend_df[0,:]
    rssta_series = low_pass(1.0/5.0,ssta_series,axis=0) 
    ax1.plot(xtime, rssta_series, 'black', alpha=1.00, linewidth=2)
    ax1.fill_between(xtime, 0., ssta_series, ssta_series> 0., color='red',   alpha=.75)
    ax1.fill_between(xtime, 0., ssta_series, ssta_series< 0., color='blue',  alpha=.75)
    #ax1.xaxis.tick_top()
    ax1.set_title("Raw trend index {}({})".format(var,vunt), fontsize=fontsize*1.1)
    ax1.set_xlabel("years", fontsize=fontsize*1.1)
    ax1.set_ylabel("degC", fontsize=fontsize*1.1)
    ax1.legend(['5-month running mean'])
    ax1.set_ylim(-4, 4)
    ax1.set_xlim(0,len(time))
    ax1.set_xticks(xtick[::120])
    ax1.set_xticklabels(xlabs[::120].astype(int))
    ax1.tick_params(labelsize=fontsize)
    ax1.grid(True)
  
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
          attrs=dict(description="trend regression maps",
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

def draw_trend_map(out_path,fig_path,da,trend_df,colnams,trend_region,mip,exp,relm,case,case_id,period,var,vunt):
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

  #ssta_series  = trend_df['trend_idx(degC)']  #.reshape(len(time))
  #rssta_series = trend_df['rtrend_idx(degC)'] #.reshape(len(time))
  #trend_series  = trend_df['trendSD_idx(1)']   #.reshape(len(time))
  #rtrend_series = trend_df['rtrendSD_idx(1)']  #.reshape(len(time))
  time  = trend_df['time']
  xtime = np.linspace(1,len(time),len(time))
  trend_df = trend_df.set_index('time') 

  trend_exist = False 
  for col in colnams: 
    if 'trend_idx' in col: 
      trend = trend_df[col].to_xarray()
      trend_exist = True
  if not trend_exist:
    exit('trend_idx not exist, please check....')

  trend    = trend.assign_coords({"time": da.time})
  trendSD  = trend/trend.std(dim='time')

  years   = int(time[0].split("-")[0])
  yeare   = int(time[len(time)-1].split("-")[0])

  # -- filtering (or Running mean)
  #rtrendSD    = trendSD.rolling(time=5, center=True).mean('time')
  #ranm       = anm.rolling(time=5, center=True).mean('time')
  rtrendSD     = trendSD 
  ranm        = anm 
  rtrendSD[:]  = low_pass(1.0/5.0,trendSD,axis=0) #5-month: 1/5.0
  if np.isnan(anm).any():
    tmp1 = ranm.fillna(-99999)
    tmp1 = low_pass(1.0/5.0,tmp1,axis=0)
    ranm = anm
    ranm[:,:,:] = tmp1[:,:,:]
    ranm = ranm.where(ranm > -10000)
    del(tmp1)
  else:
    ranm[:,:,:] = low_pass(1.0/5.0,anm[:,:,:],axis=0)
  rdanm       = detrend_dim(ranm,'time',1)
  #print(np.nanmax(ranm),np.nanmin(ranm))
  #print(np.nanmax(rdanm),np.nanmin(rdanm))

  # simultaneous
  #vreg,pval = pearsonr(rtrendSD,rdanm,axis=0)
  vcor = xr.corr(rtrendSD, rdanm, dim="time")
  vreg = xr.cov(rtrendSD, rdanm, dim="time")/rtrendSD.var(dim='time',skipna=True).values
  pval = xs.pearson_r_p_value(rtrendSD, rdanm, dim="time",skipna=True)
  
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
  ax1, fill1 = draw_regression_map(vstr,vunt,lons,lats,vcor,vreg,pval,fig,trend_region,
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
  #df2 = trend_df[ trend_df['time'] == str(da_2D.time.values)[0:10]]
  #if len(df2) > 0:
  #  ax.plot(df2['lon'], df2['lat'], 'mx', transform=ccrs.PlateCarree() )
  
  #add a box for trend region 
  draw_regional_box(region)
  #draw_screen_poly(trend_region,m)

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
  plt.savefig(os.path.join(fig_path, "trend_region_{}.pdf".format(case)))
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

def find_regional_data(da,region,dclim,vclim,l_prescribe_clim,period,pclimo,season,mip,exp,relm,case,case_id,out_path):
  #compute climatology 
  if l_prescribe_clim:
    ds = xr.open_dataset(dclim)
    clm = ds[vclim]
    del(ds)
  else: 
    #derive the climatology   
    if pclimo != period:
      ymds = '{}-{}-01'.format(pclimo.split("-")[0][0:4],pclimo.split("-")[0][4:6])
      ymde = '{}-{}-31'.format(pclimo.split("-")[1][0:4],pclimo.split("-")[1][4:6])
      dsub = da.sel(time=slice(ymds,ymde))
      clm  = dsub.groupby('time.month').mean(dim='time')
      del(ymds,ymde,dsub)
    else:
      clm  = da.groupby('time.month').mean(dim='time')
    #save clim data for future use 
    fclim = '{}.{}.{}.{}.climo.{}.{}.nc'.format(mip,exp,case,relm,case_id,var,pclimo)
    if os.path.exists(os.path.join(out_path,fclim)):
      os.remove(os.path.join(out_path,fclim))
    if not os.path.exists(out_path):
      os.makedirs(out_path)
    clm.to_netcdf(os.path.join(out_path,fclim))

  #derive anomaly 
  anm  = (da.groupby('time.month') - clm)

  lons, lats = da.longitude, da.latitude
  #Adjust lon values to facillitate the extraction of regional data
  latN = region['north']
  latS = region['south']
  lonE = region['east']
  lonW = region['west']
  if ( ((lonW < 0) or (lonE < 0 )) and (lons.values.min() > -1) ):
     da = da.assign_coords(longitude=((lons + 180) % 360 - 180) )
     lons = ( (lons + 180) % 360 - 180)
  print("range of latitude and longitude = {}-{}, {}-{}".format(min(lons.values),max(lons.values),min(lats.values),max(lats.values)))
  
  iplat = lats.where( (lats >= latS ) & (lats <= latN), drop=True)
  iplon = lons.where( (lons >= lonW ) & (lons <= lonE), drop=True)
  # -- extract data in target region
  df0 = anm.sel(latitude=iplat,longitude=iplon)
  
  if season == "Monthly": 
    df = df0  
  else:
    # == seasonal mean
    anmS = df0.rolling(time=3, center=True).mean('time')
    if season == "DJF": 
      ymds = '{}-02-01'.format(period.split("-")[0][0:4])
      ymde = '{}-12-31'.format(period.split("-")[1][0:4])
    elif season == "MAM":
      ymds = '{}-05-01'.format(period.split("-")[0][0:4])
      ymde = '{}-12-31'.format(period.split("-")[1][0:4])  
    elif season == "JJA":
      ymds = '{}-08-01'.format(period.split("-")[0][0:4])
      ymde = '{}-12-31'.format(period.split("-")[1][0:4])
    elif season == "SON":
      ymds = '{}-11-01'.format(period.split("-")[0][0:4])
      ymde = '{}-12-31'.format(period.split("-")[1][0:4])
    df = anmS.sel(time=slice(ymds,ymde,12))
    del(anmS,ymds,ymde)

  del(iplat,iplon,clm,anm,da,df0)

  return df

def trend_analysis(da,mip,exp,relm,case,case_id,period,out_path,fig_path):
  '''
  perform trend analysis on selected quantity 
  '''
  var = da.name
  vstr = da.name.upper()
  vunt = da.units

  #years = list(da.time.dt.year.data)
  times = da.time.dt.strftime("%Y-%m-%d")
  time_str = da.time.dt.strftime("%Y-%m-%d")
  ntime = len(times)
  print("number of total months in data: ", ntime)
  ntrend = 15

  lons,lats = da.longitude, da.latitude
  
  # -- EOF --
  da_detrend = detrend_dim(da,'time',1)
  da_detrend = da_detrend.sortby("latitude", ascending=True)
  da_detrend.attrs  = da.attrs 
  da_detrend.attrs['name'] = var 
  da_detrend.attrs['units'] = vunt 

  clat = da_detrend['latitude'].astype(np.float64)
  clat = np.sqrt(np.cos(np.deg2rad(clat)))
  
  wanm = da_detrend.copy()
  wanm = da_detrend * clat
  wanm.attrs = da_detrend.attrs
  wanm.attrs['long_name'] = 'Wgt: ' + da.name
  xw_anm = wanm.transpose('time', 'latitude', 'longitude')
  
  #print("data min/max (raw): {} {}".format(da_detrend.min(),da_detrend.max()))
  #print("data min/max (weighted): {} {}".format(wanm.min(),wanm.max()))
  #print("data min/max (transpose): {} {}".format(xw_anm.min(),xw_anm.max()))

  trends = trendunc_trends(xw_anm.data, ntrends=ntrend, meta=True)
  rpcs = trendunc_pcs(xw_anm.data, npcs=ntrend, meta=True)
  pcs = trendunc_pcs(xw_anm.data, npcs=ntrend, meta=True)
  pcs = pcs / pcs.std(dim='time')
  pcs['time'] = da_detrend['time']
  pcs.attrs['varianceFraction'] = trends.attrs['varianceFraction']
  #print(pcs)
  
  #evec = xr.DataArray(data=trends, dims=('trend','latitude','longitude'),
  #coords = {'trend': np.arange(0,ntrend), 
  #          'lat': xw_anm['latitude'], 
  #          'lon': xw_anm['longitude']} )
  #print(evec)
  #print(trends.attrs['varianceFraction'])

  do = xr.Dataset(coords = {
                  'time': da_detrend['time'],
                  'trend': np.arange(0,ntrend), 
                  'lat': xw_anm['latitude'], 
                  'lon': xw_anm['longitude']
                  },
                  attrs={'var_in': var,
                         'var_name': vstr,
                         'units': vunt},
                  )
  do['trends']   = (['trend','lat','lon'],trends.data)
  do['pcs']    = (['trend','time'],pcs.data)
  do['rawpcs'] = (['trend','time'],rpcs.data)
  do['varianceFraction'] = (['trend'],trends.attrs['varianceFraction'].data)
  if not os.path.exists(out_path):
    os.makedirs(out_path)
  fout_name = "{}.{}.{}.{}.{}.{}.trend.{}.{}.nc".format(mip,exp,case,relm,case_id,vstr,regnam,period)
  do.to_netcdf(os.path.join(out_path,fout_name)) 
  do.close()

  del(lons,lats,times,time_str,clat,wanm,xw_anm,trends,rpcs,do)

  return pcs,da_detrend

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
  out_path = os.path.join(top_path,"E3SMv21_testings","paper_material","fig_data","trend_analysis","raw_index")
  fig_path = os.path.join(top_path,"E3SMv21_testings","paper_material","8_trend_analysis","trend_index_ts","figure")

  # region of interest (trend sector)
  trend_region = {'socn':{'west':0., 'east':360., 'south':-90., 'north':-45.}}

  mip       = 'analysis'
  exp       = 'historical'
  relm      = 'en00'
  case_id   = 'trend_gwmsl'
  tableId   = "Amon"
  products  = ["NOAA_20C","ERA5"]
  varlist   = ["ts", "tas"]
  varstr    = ["TS", "TREFHT"]
  period    = "195001-201412"
  pclimo    = "195001-198012"
  run_path  = "/lcrc/group/e3sm/ac.szhang/acme_scratch/data/pcmdi/model/monthly"
  run_mask  = "/lcrc/group/e3sm/ac.szhang/acme_scratch/data/pcmdi/model/fixed/sftlf"
  season    = "Monthly"
  for k,regnam in enumerate(trend_region.keys()):
    region = trend_region[regnam]
    for j,var in enumerate(varlist): 
      for i,product in enumerate(products):
        case = product
        ptmp = os.path.join(run_path,mip,case,tableId,var)
        ftmp = '{}.{}.{}.{}.*.{}.{}.nc'.format(mip,exp,case,relm,var,"*")
        clim_tmp = '{}.{}.{}.{}.climo.{}.{}.nc'.format(mip,exp,case,relm,case_id,var,pclimo)
        filePath = sorted(glob.glob(os.path.join(ptmp,ftmp)))
        if len(filePath) > 0 and os.path.isfile(filePath[0]):
          data_fil = filePath[0]
          clim_fil = os.path.join(out_path,clim_tmp)
          mask_fil = os.path.join(run_mask,tableId,mip+"."+exp+"."+case+".fx.sftlf.nc")
          case_dict = collections.OrderedDict()
          if case == "ERA5_HRES": 
            case_dict[case] = Case(data_fil, var=var, color="blue", label=case)
          else:
            case_dict[case] = Case(data_fil, var=varstr[j], color="blue", label=case)
          case_dict['mask'] = Case(mask_fil, var="sftlf", color="blue", label=case)
          case_dict['clim'] = Case(clim_fil, var=varstr[j], color="blue", label=case)
          #call fuction to generate rgmn index 
          main(fig_path,out_path,mip,exp,relm,case_id,period,pclimo,case_dict,region,regnam,season)
