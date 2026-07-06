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

def main(fig_path,out_path,mip,exp,relm,case_id,period,case_dict,nino_region,l_check_nino_region):
  for key in case_dict:
    if key == "mask":
      dmsk = case_dict[key].path
      vmsk = case_dict[key]._var
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
  da = ds[var].where(mask == 0)
  if da.units == "K":
    # change units
    print("change units: from ", da.units, " to ", "degC")
    da = da - 273.15
    da = da.assign_attrs(units='^{o}C')
  
  ################################
  #sanity check on nino region
  ################################
  if l_check_nino_region:
    #only select first year to check  
    da_t = ds[var].sel(time=ds[var].time[0:11]) #.dt.year.values[0])
    #Select region and apply land-sea mask
    da_mask = da_t.where(mask == 0)
    ### slice area around ASL region
    da_nmsk = slice_region(da_t, nino_region)
    da_mask = slice_region(da_mask, nino_region)
    print(da_mask)
    print("before mask,min/max: ", da.min(),da.max())
    print("after mask,min/max: ",da_mask.min(),da_mask.max())
    plot_regions_mask(fig_path,da_nmsk,da_mask,case)
    del(da_t,da_mask,da_nmsk)
  ################################
  #end sanity check on nino region
  ################################
  var = da.name 
  vstr = da.name.upper()
  vunt = da.units

  #calculate and save nino index 
  nino_df,colnams = define_nino(mip,exp,relm,case,case_id,period,vstr,vunt,da,nino_region,out_path)

  #plot nino time series 
  draw_nino_ts(fig_path,nino_df,colnams,nino_region,mip,exp,relm,case,case_id,period,vstr,vunt) 

  #plot map with nino regression on TS
  draw_nino_map(out_path,fig_path,ds[var],nino_df,colnams,nino_region,mip,exp,relm,case,case_id,period,vstr,vunt)
  
  return

# -- Detrending
def detrend_dim(da, dim, deg=1):
    # detrend along a single dimension
    p = da.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    return da - fit

def draw_nino_ts(fig_path,nino_df,colnams,nino_region,mip,exp,relm,case,case_id,period,var,vunt): 

  time  = nino_df['time']
  xtime = np.linspace(1,len(time),len(time))
  for col in colnams:
    print(col)
    if 'nino_idx' in col: 
      ssta_series  = np.array(nino_df[col])  #.reshape(len(time))
    if 'rnino_idx' in col: 
      rssta_series = np.array(nino_df[col]) #.reshape(len(time))
    if 'ninoSD_idx' in col: 
      nino_series = np.array(nino_df[col]) #.reshape(len(time))
    if 'rninoSD_idx' in col: 
      rnino_series = np.array(nino_df[col]) #.reshape(len(time))
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
  ax1.set_title("Raw nino index {}({})".format(var,vunt), fontsize=fontsize*1.1)
  ax1.set_xlabel("years", fontsize=fontsize*1.1)
  ax1.set_ylabel("degC", fontsize=fontsize*1.1)
  ax1.legend(['5-month running mean'])
  ax1.set_ylim(-4, 4)
  ax1.set_xlim(0,len(time))
  ax1.set_xticks(xtick[::120])
  ax1.set_xticklabels(xlabs[::120].astype(int))
  ax1.tick_params(labelsize=fontsize)
  ax1.grid(True)

  ax2.plot(xtime, rnino_series, 'black', alpha=1.00, linewidth=2)
  ax2.fill_between(xtime, 0., nino_series, nino_series> 0., color='red',   alpha=.75)
  ax2.fill_between(xtime, 0., nino_series, nino_series< 0., color='blue',  alpha=.75)
  #ax2.xaxis.tick_top()
  ax2.set_title("Standardized nino index {}({})".format(var,vunt), fontsize=fontsize*1.1)
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
          attrs=dict(description="nino regression maps",
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

def draw_nino_map(out_path,fig_path,da,nino_df,colnams,nino_region,mip,exp,relm,case,case_id,period,var,vunt):
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

  #ssta_series  = nino_df['nino_idx(degC)']  #.reshape(len(time))
  #rssta_series = nino_df['rnino_idx(degC)'] #.reshape(len(time))
  #nino_series  = nino_df['ninoSD_idx(1)']   #.reshape(len(time))
  #rnino_series = nino_df['rninoSD_idx(1)']  #.reshape(len(time))
  time  = nino_df['time']
  xtime = np.linspace(1,len(time),len(time))
  nino_df = nino_df.set_index('time') 

  nino_exist = False 
  for col in colnams: 
    if 'nino_idx' in col: 
      nino = nino_df[col].to_xarray()
      nino_exist = True
  if not nino_exist:
    exit('nino_idx not exist, please check....')

  nino    = nino.assign_coords({"time": da.time})
  ninoSD  = nino/nino.std(dim='time')

  years   = int(time[0].split("-")[0])
  yeare   = int(time[len(time)-1].split("-")[0])

  # -- filtering (or Running mean)
  #rninoSD    = ninoSD.rolling(time=5, center=True).mean('time')
  #ranm       = anm.rolling(time=5, center=True).mean('time')
  rninoSD     = ninoSD 
  ranm        = anm 
  rninoSD[:]  = low_pass(1.0/5.0,ninoSD,axis=0) #5-month: 1/5.0
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
  #vreg,pval = pearsonr(rninoSD,rdanm,axis=0)
  vcor = xr.corr(rninoSD, rdanm, dim="time")
  vreg = xr.cov(rninoSD, rdanm, dim="time")/rninoSD.var(dim='time',skipna=True).values
  pval = xs.pearson_r_p_value(rninoSD, rdanm, dim="time",skipna=True)
  
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
  ax1, fill1 = draw_regression_map(vstr,vunt,lons,lats,vcor,vreg,pval,fig,nino_region,
                                   fontsize,'Nino3.4-SST Pattern',grid[0,0],vmin,vmax,nlev)
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
  #df2 = nino_df[ nino_df['time'] == str(da_2D.time.values)[0:10]]
  #if len(df2) > 0:
  #  ax.plot(df2['lon'], df2['lat'], 'mx', transform=ccrs.PlateCarree() )
  
  #add a box for nino region 
  draw_regional_box(region)
  #draw_screen_poly(nino_region,m)

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
  plt.savefig(os.path.join(fig_path, "nino_region_{}.pdf".format(case)))
  plt.close()
  return

def low_pass(cutoff_freq, data, order=5, axis=-1):
    #low-pass: filtering high-frequency signal
    #nyquist normalized cutoff for digital design
    Wn = cutoff_freq
    b, a = butter(order, Wn, btype='lowpass', analog=False)
    data_filt = filtfilt(b, a, data, axis=axis, method="gust")
    return data_filt

def define_nino(mip,exp,relm,case,case_id,period,vstr,vunt,da,region,out_path):
  '''
  da for one point in time (with lats x lons)
  '''
  #years = list(da.time.dt.year.data)
  times = da.time.dt.strftime("%Y-%m-%d")
  time_str = da.time.dt.strftime("%Y-%m-%d")
  ntime = len(times)
  print("number of total months in data: ", ntime)

  clm  = da.groupby('time.month').mean(dim='time')
  anm  = (da.groupby('time.month') - clm)
  dclm = anm.copy()
  for i in range(len(clm)):
    dclm[i::12,:,:] = clm[i,:,:].copy()

  nino_sst    = wgt_areaave(da,region)
  nino_clm    = wgt_areaave(dclm,region)
  nino_ind    = wgt_areaave(anm,region)
  ninoSD_ind  = nino_ind / nino_ind.std(dim='time')
  #instead of moving average, using a low-pass filtering
  #rnino_ind   = nino_ind.rolling(time=5, center=True).mean('time')
  #onino_ind   = nino_ind.rolling(time=3, center=True).mean('time')
  #rninoSD_ind = ninoSD_ind.rolling(time=5, center=True).mean('time')
  #oninoSD_ind = ninoSD_ind.rolling(time=3, center=True).mean('time')
  rnino_ind    = low_pass(1.0/5.0,nino_ind,axis=0)   #5-month: 1/5.0
  onino_ind    = low_pass(1.0/3.0,nino_ind,axis=0)   #3-month: 1/3.0
  rninoSD_ind  = low_pass(1.0/5.0,ninoSD_ind,axis=0) #5-month: 1/5.0
  oninoSD_ind  = low_pass(1.0/3.0,ninoSD_ind,axis=0) #3-month: 1/3.0

  colnams = ['time','ninosst_val(degC)','ninosst_clm(degC)','nino_idx(degC)','rnino_idx(degC)',
             'onino_idx(degC)','ninoSD_idx(1)','rninoSD_idx(1)','oninoSD_idx(1)']
  df = pd.DataFrame([],columns=colnams)
  for col in colnams: 
    if 'time' in col: 
      df[col] = time_str
    elif 'ninosst_val' in col: 
      df[col] = nino_sst
    elif 'ninosst_clm' in col:
      df[col] = nino_clm
    elif 'nino_idx' in col:
      df[col] = nino_ind
    elif 'ninoSD_idx' in col:
      df[col] = ninoSD_ind
    elif 'rnino_idx' in col:
      df[col] = rnino_ind
    elif 'rninoSD_idx' in col:
      df[col] = rninoSD_ind
    elif 'onino_idx' in col:
      df[col] = onino_ind
    elif 'oninoSD_idx' in col:
      df[col] = oninoSD_ind
  ### clean-up DataFrame
  df = df.reset_index(drop=True)
  #save data to text file
  if not os.path.exists(out_path):
    os.makedirs(out_path)
  out_file = '{}.{}.{}.{}.{}.{}.{}.csv'.format(mip,exp,case,relm,case_id,vstr,period)
  df.to_csv(os.path.join(out_path,out_file),index=False)

  del(clm,anm,dclm,da,nino_sst,nino_clm,nino_ind,rnino_ind,ninoSD_ind,rninoSD_ind,times,time_str)

  return df,colnams 

def wgt_areaave(da,region):
  lons, lats = da.longitude, da.latitude
  #Adjust lon values to facillitate the regional mean calculation
  latN = region['north']
  latS = region['south']
  lonE = region['east']
  lonW = region['west']
  print("range of latitude and longitude = {}-{}, {}-{}".format(min(lons.values),max(lons.values),min(lats.values),max(lats.values)))
  if ( ((lonW < 0) or (lonE < 0 )) and (lons.values.min() > -1) ):
     da = da.assign_coords(longitude=((lons + 180) % 360 - 180) )
     lons = ( (lons + 180) % 360 - 180)
  #lon_name = "longitude"
  #ds['_longitude_adjusted'] = xr.where(ds[lon_name] > 180,ds[lon_name] - 360,ds[lon_name])
  #ds = (ds.swap_dims({lon_name: '_longitude_adjusted'})
  #        .sel(**{'_longitude_adjusted': sorted(ds._longitude_adjusted)})
  #        .drop(lon_name))
  #ds = ds.rename({'_longitude_adjusted': lon_name})

  iplat = lats.where( (lats >= latS ) & (lats <= latN), drop=True)
  iplon = lons.where( (lons >= lonW ) & (lons <= lonE), drop=True)

  wgt  = np.cos(np.deg2rad(lats))
  odat = da.sel(latitude=iplat,longitude=iplon).weighted(wgt).mean(("longitude", "latitude"), skipna=True)
  del(lats,lons,da,iplat,iplon,wgt)

  return odat

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
  out_path = os.path.join("/lcrc/group/e3sm/public_html/diagnostic_output/ac.szhang/polar_analysis/data","nino_analysis","raw_index")
  fig_path = os.path.join("/lcrc/group/e3sm/public_html/diagnostic_output/ac.szhang/polar_analysis/figure", "nino_analysis","nino_index_ts","figure")

  # region of interest (nino sector)
  nino_region   = {'west':-170., 'east':-120., 'south':-5., 'north':5.}

  #sanity check (plot region with and without mask) 
  l_check_nino_region = False #True  

  mip       = 'analysis'
  exp       = 'historical'
  relm      = 'en00'
  case_id   = 'nino34'
  tableId   = "Amon"
  products  = ["NOAA_20C","ERA5"]
  var       = "ts"
  varstr    = "TS"
  period    = "195001-201412"
  run_path  = "/lcrc/group/e3sm/ac.szhang/acme_scratch/data/pcmdi/model/monthly"
  run_mask  = "/lcrc/group/e3sm/ac.szhang/acme_scratch/data/pcmdi/model/fixed/sftlf"
  for i,product in enumerate(products):
    case = product
    ptmp = os.path.join(run_path,mip,case,tableId,var)
    ftmp = '{}.{}.{}.{}.*.{}.{}.nc'.format(mip,exp,product,relm,var,"*")
    filePath = sorted(glob.glob(os.path.join(ptmp,ftmp)))
    if len(filePath) > 0 and os.path.isfile(filePath[0]):
      data_fil = filePath[0]
      print(data_fil)
      mask_fil = os.path.join(run_mask,tableId,mip+"."+exp+"."+case+".fx.sftlf.nc")
      case_dict = collections.OrderedDict()
      if case == "ERA5_HRES": 
        case_dict[case] = Case(data_fil, var=var, color="blue", label=case)
      else:
        case_dict[case] = Case(data_fil, var=varstr, color="blue", label=case)
      case_dict['mask'] = Case(mask_fil, var="sftlf", color="blue", label=case)
      #call fuction to generate rgmn index 
      main(fig_path,out_path,mip,exp,relm,case_id,period,case_dict,nino_region,l_check_nino_region)
