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
import matplotlib.ticker as mticker
import matplotlib.path as mpath
import matplotlib.patheffects as PathEffects
from matplotlib.pylab import rcParams
from matplotlib.patches import Polygon

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

from scipy import stats
from scipy.stats import pearsonr
from scipy.signal import butter, filtfilt, sosfilt,lfilter
from mpl_toolkits.basemap import Basemap

import cmaps as gvcmaps
import geocat.viz.util as gvutil
import geocat.viz as gv
import cmaps as gvcmaps
from geocat.comp import eofunc_eofs, eofunc_pcs
from global_land_mask import globe

def main(fig_path,out_path,mip,exp,relm,case_id,period,pclimo,case_dict,region,regnam,season):  
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
  if (var == 'TS'):
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

  #derive data for eof analysis
  if os.path.exists(dclim):
    l_prescribe_clim = True 
  else:
    l_prescribe_clim = False
  df = find_regional_data(da,region,regnam,dclim,vclim,l_prescribe_clim,period,pclimo,season,mip,exp,relm,case,case_id,out_path)
  df.attrs['name'] = da.name
  df.attrs['units'] = da.units

  #calculate and save eof index 
  eof_analysis(df,season,regnam,mip,exp,relm,case,case_id,period,out_path,fig_path)

  #plot eof time series
  draw_eof_ts(df,season,regnam,mip,exp,relm,case,case_id,period,out_path,fig_path)
  
  #plot map with eof regression on TS
  draw_eof_map(df,season,region,regnam,mip,exp,relm,case,case_id,period,out_path,fig_path)
  
  return

# -- Detrending
def detrend_dim(da, dim, deg=1):
    # detrend along a single dimension
    p = da.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    return da - fit

def draw_eof_ts(da,season,regnam,mip,exp,relm,case,case_id,period,out_path,fig_path): 
  var = da.attrs['name']
  vstr = da.attrs['name'].upper()
  vunt = da.attrs['units']
  
  #years = list(da.time.dt.year.data)
  time = da['time'] #.time.dt.strftime("%Y-%m-%d")
  xtime = np.linspace(1,len(time),len(time))
  years = int(time.dt.year.min())
  yeare = int(time.dt.year.max())

  # -- figure plot
  for typ in [ "raw", "detrend"]: 
    #load data file 
    fname = "{}.{}.{}.{}.{}.{}.eof.{}.{}.{}.{}.nc".format(
             mip,exp,case,relm,case_id,vstr,typ,season,regnam,period)
    df = xr.open_dataset(os.path.join(out_path,fname))
    if season == "Monthly":
      xtick = np.arange(0,len(time))
      xlabs = np.arange(0,len(time)) / 12.0  + years
      nint  = 240
      legend = ['5-month running mean']
      ylabel = "Standardized index"
      xlabel = "Time (year/month)"
    else:
      xtick = np.arange(0,len(time))
      xlabs = np.arange(0,len(time)) + years
      nint  = 10
      legend = ['5-year running mean']
      ylabel = "Standardized index"
      xlabel = "Time (year)"
    fig, ax = plt.subplots(5, 1, figsize=(12, 16))
    fontsize = 16

    for ieof in range(0,5):
      pct = df['varianceFraction'][ieof] * 100
      if typ == 'raw':
        title = "EOF{} PCS of {}({:.2f}%)".format(ieof+1,vstr,pct)
      else:
        title = "EOF{} PCS of Detrended {}({:.2f}%)".format(ieof+1,vstr,pct)
      ssta_series = df['pcs'][ieof,:]
      rssta_series = low_pass(1.0/5.0,ssta_series,axis=0)
      colors = ['blue' if val < 0 else 'red' for val in ssta_series]
      ax[ieof].plot(xtime, rssta_series, 'black', alpha=1.00, linewidth=2)
      #ax[ieof].fill_between(xtime, 0., ssta_series, ssta_series> 0., color='red',   alpha=.75)
      #ax[ieof].fill_between(xtime, 0., ssta_series, ssta_series< 0., color='blue',  alpha=.75)
      ax[ieof].bar(xtime,ssta_series,color=colors,width=1.0,edgecolor='black',alpha=.75,linewidth=0.1)
      #ax[ieof].xaxis.tick_top()
      ax[ieof].set_title(title, fontsize=fontsize*1.1)
      ax[ieof].set_xlabel(xlabel, fontsize=fontsize*1.1)
      ax[ieof].set_ylabel(ylabel, fontsize=fontsize*1.1)
      ax[ieof].legend(legend)
      ax[ieof].set_ylim(-4, 4)
      ax[ieof].set_xlim(0,len(time))
      ax[ieof].set_xticks(xtick[::nint])
      ax[ieof].set_xticklabels(xlabs[::nint].astype(int))
      ax[ieof].tick_params(labelsize=fontsize)
      ax[ieof].grid(True)
  
    #plt.draw()
    plt.tight_layout()

    if not os.path.exists(fig_path):
      os.makedirs(fig_path)
    fig_name = "fig_eofts_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.pdf".format(
                typ,mip,exp,case,relm,case_id,vstr,season,regnam,period)
    plt.savefig(os.path.join(fig_path, fig_name))
    plt.close()
    del(fname,df,xtick,xlabs,fig,ax)

  return

def save_figure_data(season,regnam,typ,ieof,vcor,vreg,pval,vsig,
                     mip,exp,relm,case,case_id,period,var,vunt,out_path):
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
          attrs=dict(description="eof regression maps",
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
  fout_name = "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_eof{:02d}_regression.nc".format(
               mip,exp,case,relm,case_id,var,season,regnam,period,typ,ieof+1)
  df.to_netcdf(os.path.join(out_path,fout_name))

  return 

def draw_eof_map(da,season,region,regnam,mip,exp,relm,case,case_id,period,out_path,fig_path):
  var = da.attrs['name']
  vstr = da.attrs['name'].upper()
  vunt = da.attrs['units']
  neofsub = 5

  #years = list(da.time.dt.year.data)
  time = da['time'] #.dt.strftime("%Y-%m-%d")
  xtime = np.linspace(1,len(time),len(time))
  years = int(time.dt.year.min())
  yeare = int(time.dt.year.max())
  
  lats, lons = da.latitude, da.longitude 

  # -- figure plot
  for typ in [ "raw", "detrend"]:
    #load data file
    fname = "{}.{}.{}.{}.{}.{}.eof.{}.{}.{}.{}.nc".format(
             mip,exp,case,relm,case_id,vstr,typ,season,regnam,period)
    df = xr.open_dataset(os.path.join(out_path,fname))
    #generate figure
    fig = plt.figure(figsize=(20, 8))
    grid = fig.add_gridspec(ncols=neofsub,nrows=1,wspace=0.4,hspace=0.2)
    vmin = -2.0
    vmax =  2.0
    nlev =  21
    fontsize = 14
    bartitle = 'Regressed {}({})'.format(vstr,vunt)
    axs = []
    fills = []
    for ieof in range(neofsub):
      eofSD = df['pcs'][ieof,:]
      # -- filtering (or Running mean)
      ##reofSD = eofSD.rolling(time=5, center=True).mean('time')
      ##ranm = anm.rolling(time=5, center=True).mean('time')
      #reofSD = eofSD
      #ranm = anm
      #reofSD[:] = low_pass(1.0/5.0,eofSD,axis=0) #5-month: 1/5.0
      #if np.isnan(anm).any():
      #  tmp1 = ranm.fillna(-99999)
      #  tmp1 = low_pass(1.0/5.0,tmp1,axis=0)
      #  ranm = anm
      #  ranm[:,:,:] = tmp1[:,:,:]
      #  ranm = ranm.where(ranm > -10000)
      #  del(tmp1)
      #else:
      #  ranm[:,:,:] = low_pass(1.0/5.0,anm[:,:,:],axis=0)
      #rdanm = detrend_dim(ranm,'time',1)
      ##print(np.nanmax(ranm),np.nanmin(ranm))
      ##print(np.nanmax(rdanm),np.nanmin(rdanm))
      if typ == 'raw':
        title = "EOF{} PCS for {}({})".format(ieof+1,vstr,vunt)
        ylabel = "Standardized index"
        xlabel = "Time (year/month)"
        da_detrend = da
      else:
        title = "EOF{} PCS for {}({}) with Detrend".format(ieof+1,vstr,vunt)
        ylabel = "Standardized index"
        xlabel = "Time (year/month)"
        da_detrend = detrend_dim(da,'time',1)

      da_detrend = da_detrend.sortby("latitude", ascending=True)
      da_detrend.attrs = da.attrs
      da_detrend.attrs['name'] = var
      da_detrend.attrs['units'] = vunt
      da_detrend = da_detrend.transpose('time', 'latitude', 'longitude')

      # simultaneous
      #vreg,pval = pearsonr(reofSD,rdanm,axis=0)
      vcor = xr.corr(eofSD, da_detrend, dim="time")
      vreg = xr.cov(eofSD, da_detrend, dim="time")/eofSD.var(dim='time',skipna=True).values
      pval = xs.pearson_r_p_value(eofSD, da_detrend, dim="time",skipna=True)
      #correlation at 5% significant level
      vsig = vreg.copy() 
      #vsig = np.ma.masked_array(vreg, mask=(pval>=0.05)) #np.where(pval >= 0.05,np.nan,vreg)
      vsig = vsig.where(pval<=0.05)
      save_figure_data(season,regnam,typ,ieof,vcor,vreg,pval,vsig,
                       mip,exp,relm,case,case_id,period,var,vunt,out_path)

      # Show the plot
      ax1, fill1 = draw_regression_map(vcor,vreg,pval,fig,region,regnam,fontsize,
                                       'EOF{} on {}'.format(ieof+1,vstr),
                                       grid[ieof],vmin,vmax,nlev)
      axs.append(ax1)
      fills.append(fill1)
      del(ax1,fill1)

    cbar = fig.colorbar(fills[neofsub-1],
                 ax=axs,
                 drawedges=True,
                 orientation='horizontal',
                 ticks = [vmin + (vmax-vmin)*i/(nlev-1) for i in range(nlev)],
                 label='', #'Regressed {}({})'.format(vstr,vunt),
                 shrink=0.95,
                 aspect=80,
                 pad=0.15,
                 extendfrac='auto',
                 extendrect=True)
    cbar.ax.tick_params(labelsize=fontsize*0.9)
    cbar.set_label(label='Regressed {}({})'.format(vstr,vunt),fontsize=fontsize*0.95)
   #cbar.set_ticks(ticks=ticks,labels=labels,fontsize=fontsize*0.9)  

    #fig.suptitle('SST correlation & regression with Nino3.4', fontsize=fontsize, y=0.9)
    #plt.rcParams["font.family"] = "sans-serif"
    #plt.rcParams.update({'font.size': fontsize})
    plt.draw()

    if not os.path.exists(fig_path):
      os.makedirs(fig_path)
    fig_name = "fig_eofreg_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.pdf".format(
                typ,mip,exp,case,relm,case_id,vstr,season,regnam,period)
    plt.savefig(os.path.join(fig_path, fig_name))
    plt.close()
   
  return

def draw_regression_map(cor,reg,pval,fig,region,regnam,font,title,grid_space,vmin,vmax,nlev):
  sig    = pval
  sig[:] = 1.0 - sig[:]
  t90    = 0.94
  t95    = 0.95
  llabel = ''
  rlabel = '' #'{}({})'.format(vstr,vunt)
  
  #extract latitude and logitude info
  lons,lats = reg.longitude, reg.latitude
   
  #proj = ccrs.PlateCarree(central_longitude=210)
  proj = ccrs.SouthPolarStereo()
  #kwtrans = dict(central_latitude=-90, central_longitude=0.,true_scale_latitude=None, globe=None)
  #trans = ccrs.Stereographic(**kwtrans)

  allowed_plotstyles = ['pcolormesh', 'contour', 'contourf']
  use_plotstyle = 'contourf' #'pcolormesh'
  # To prevent the line wrapping for data on a curvilinear grid using contour or
  # contourf plot style, we have to compute the data transformation to the
  # NorthPolarStereo projection. This takes some time.
  # For curvilinear grid uncomment code below
  #if use_plotstyle == 'contour' or use_plotstyle == 'contourf':
  #  if 'SouthPolar' in str(proj): 
  #    x, y = transform_to_proj(reg.longitude, reg.latitude, "SH")
  #  else:
  #    x, y = transform_to_proj(reg.longitude, reg.latitude, "NH")  
  if use_plotstyle == 'contour' or use_plotstyle == 'contourf': 
      x, y = np.meshgrid(lons,lats) 

  ax = fig.add_subplot(grid_space,projection=proj)
  ax.set_extent([min(lons),max(lons),min(lats),max(lats)], ccrs.PlateCarree())
  #ax.background_img(name='BlueMarbleBright', resolution='high')
  ax.coastlines(linewidth=0.5, alpha=0.6)
  #ax.set_global()

  # Draw map features on top of filled contour
  #ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=1)
  #ax.add_feature(cfeature.COASTLINE, edgecolor='gray', linewidth=0.5, zorder=1)
  ax.add_feature(cfeature.LAND, facecolor='none', zorder=1)
  ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=0.5, zorder=1)

  #-- set circular boundary for the map from Cartopy's 'Custom Boundary Shape' example
  #-- https://scitools.org.uk/cartopy/docs/latest/_downloads/b310547efca1511c5b83d11f3c6b0115/always_circular_stereo.py
  theta  = np.linspace(0, 2*np.pi, 100)
  center = [0.5, 0.5]
  radius =  0.5
  verts  = np.vstack([np.sin(theta), np.cos(theta)]).T
  circle = mpath.Path(verts * radius + center)
  ax.set_boundary(circle, transform=ax.transAxes)

  ## Use geocat.viz.util convenience function to set axes limits & tick values
  #gvutil.set_axes_limits_and_ticks(ax,
  #                               xlim=(-180, 180),
  #                               ylim=(-90, 90),
  #                               xticks=np.arange(-180, 181, 60),
  #                               yticks=np.arange(-90, 91, 30))
  #
  ## Use geocat.viz.util convenience function to add minor and major tick lines
  #gvutil.add_major_minor_ticks(ax, labelsize=font*0.90)

  ## Use geocat.viz.util convenience function to make latitude, longitude tick labels
  #gvutil.add_lat_lon_ticklabels(ax)

  # Import the default color map
  #newcmp = gvcmaps.BlueYellowRed
  #index = [5, 20,  35, 50, 65, 85, 95, 110, 125,  0, 0, 135, 150,  165, 180, 200, 210, 220, 235, 250 ]
  #color_list = [newcmp[i].colors for i in index]
  ##-- Change to white
  #color_list[9]=[ 1., 1., 1.]
  #color_list[10]=[ 1., 1., 1.]
  newcmp = gvcmaps.BlueYellowRed
  color_list = newcmp(np.linspace(0, 1, nlev+1))
  #nmid = int((nlev+1)/ 2)
  #color_list[nmid-1]=[ 1., 1., 1., 1.]
  #color_list[nmid]=[ 1., 1., 1., 1.]

  #-- Change to white
  # Define dictionary for kwargs
  levels = np.linspace(vmin,vmax,nlev+2)
  levels[1:nlev+1] = np.linspace(vmin,vmax,nlev)
  levels[0] = levels[0] - (vmax - vmin)/(nlev-1)
  levels[nlev+1] = levels[nlev] + (vmax - vmin)/(nlev-1)
  #-- create data plot
  if use_plotstyle == 'pcolormesh':
    kwargs = dict(
      vmin = vmin,
      vmax = vmax,
      transform=ccrs.PlateCarree(),
      cmap = newcmp)
    fillplot = ax.pcolormesh(reg.longitude, reg.latitude, reg.data, **kwargs)
  else:  #-- Note: in this case do not use transform keyword!
    kwargs = dict(
      vmin = vmin,
      vmax = vmax,
      levels = levels,
      colors=color_list,
      #add_colorbar=False,  # allow for colorbar specification later
      transform=ccrs.PlateCarree(),)    
    fillplot = getattr(ax,use_plotstyle)(x, y, reg.data, **kwargs)

  # Plot Hatch for significance
  sig.plot.contourf(ax=ax, levels = [-1*t95, -1*t90, t90, t95], colors='none',
      hatches=[None, None, None, '..', '..'], extend='both',
      add_colorbar=False, transform=ccrs.PlateCarree())

  # Plot line contours for correlation
  # Specify contour levels excluding 0
  #delc=0.2
  #level2 = np.arange(-1, 0, delc)
  #level2 = np.append(levels, np.arange(delc, 1, delc))
  #rad = cor.plot.contour(ax=ax,
  #                colors='black',
  #                alpha=0.8,
  #                linewidths=1.0,
  #                add_labels=False,
  #                levels=level2,
  #                transform=ccrs.PlateCarree())
  #ax.clabel(rad, levels, fmt='%1.1f',  inline=True, colors='black', fontsize=font)
  #pe = [PathEffects.withStroke(linewidth=2.0, foreground="w")]
  #plt.setp(rad.collections, path_effects=pe)

  #-- add grid lines and annotations
  gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, dms=False,
                    x_inline=False, y_inline=False, linewidth=1,
                    linestyle='dotted', color="black", alpha=0.3)
  # Manipulate latitude and longitude gridline numbers and spacing
  #gl.xlocator = mticker.FixedLocator(np.arange(-180,180,60))
  #gl.xformatter = LONGITUDE_FORMATTER
  #gl.ylocator = mticker.FixedLocator(np.arange(-90,0,15))
  #gl.yformatter = LATITUDE_FORMATTER
  #gl.xlabel_style = dict(weight='normal', color='black', size=font*0.8) #,rotation=45)
  #gl.ylabel_style = dict(weight='normal', color='black', size=font*0.8) #,rotation=45)
  #gl.bottom_labels = False
  #gl.left_labels = False
  #gl.top_labels = False
  #gl.right_labels = False
  #gl.rotate_labels = False 

  #manually setup labels
  xticks = [0, 60, 120, 180, 240, 300, 360]
  lon_formatter = LongitudeFormatter()
  lon_formatter.set_locs(xticks) 
  xtlabs = [lon_formatter(value) for value in xticks]
  yticks = np.full_like(xticks, max(lats)+1.0)  # Latitude where the labels will be drawn
  for xtick, ytick, label in zip(xticks, yticks, xtlabs):
    if xtick in [120, 300]: 
      rotation = 60;
    elif xtick in [60, 240]:  
      rotation = 120;
    else:
      rotation = 0 
    if 'W' in label:
      ax.text(xtick,ytick,label,rotation=rotation,fontsize=font*0.85,
              horizontalalignment='right',verticalalignment='center',
              transform=ccrs.Geodetic())
    elif 'E' in label:
      ax.text(xtick,ytick,label,rotation=rotation,fontsize=font*0.85,
              horizontalalignment='left',verticalalignment='center',
              transform=ccrs.Geodetic())
    elif '180' in label: 
      ax.text(xtick,ytick,label,rotation=rotation,fontsize=font*0.85,
              horizontalalignment='center',verticalalignment='top',
              transform=ccrs.Geodetic())
    else:
      ax.text(xtick,ytick,label,rotation=rotation,fontsize=font*0.85,
              horizontalalignment='center',verticalalignment='bottom',
              transform=ccrs.Geodetic())

  #yticks = [-90, -60, -30, 0, 30, 60, 90]
  #lat_formatter = LatitudeFormatter()
  #lat_formatter.set_locs(yticks)
  #ylabels = [lat_formatter(value) for value in yticks]

  #turn off ticks on top and right
  #ax.xaxis.tick_bottom()
  #ax.yaxis.tick_left()

  # Use geocat.viz.util convenience function to add titles to left and right of the plot axis.
  gvutil.set_titles_and_labels(ax,
                             maintitle=title,
                             maintitlefontsize=font,
                             lefttitle=llabel,
                             lefttitlefontsize=font*0.95,
                             righttitle=rlabel,
                             righttitlefontsize=font*0.95,
                             labelfontsize = font*0.95,
                             xlabel="",
                             ylabel="")


  # add mark for specific location 
  #df2 = eof_df[ eof_df['time'] == str(da_2D.time.values)[0:10]]
  #if len(df2) > 0:
  #  ax.plot(df2['lon'], df2['lat'], 'mx', transform=ccrs.PlateCarree() )
  
  #add a box for eof region 
  #draw_regional_box(region)
  #draw_screen_poly(region,m)

  return ax, fillplot

def transform_to_proj(longitude, latitude, polar_type):
    lat = latitude.values.ravel()
    lon = longitude.values.ravel()
    X, Y = [], []
    for itr in range(len(lat)):
      if polar_type == "NH":   
        x, y = ccrs.NorthPolarStereo().transform_point(lon[itr], lat[itr], ccrs.PlateCarree())
      else:
        x, y = ccrs.SouthPolarStereo().transform_point(lon[itr], lat[itr], ccrs.PlateCarree())
      X.append(x)
      Y.append(y)
    shapes = latitude.shape
    X = np.reshape(X, (shapes[0], shapes[1]))
    Y = np.reshape(Y, (shapes[0], shapes[1]))
    return X, Y

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
  plt.savefig(os.path.join(fig_path, "eof_region_{}.pdf".format(case)))
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

def find_regional_data(da,region,regnam,dclim,vclim,l_prescribe_clim,period,pclimo,season,mip,exp,relm,case,case_id,out_path):
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
    var = vclim.upper()
    fclim = '{}.{}.{}.{}.climo.{}.{}.{}.{}.nc'.format(mip,exp,case,relm,case_id,var,regnam,pclimo)
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
     anm = anm.assign_coords(longitude=((lons + 180) % 360 - 180) )
     anm = anm.sortby("longitude")
     print(' change longitude ')
     #lons = ( (lons + 180) % 360 - 180)
     lons = anm.longitude 
  print("range of latitude and longitude = {}-{}, {}-{}".format(min(lons.values),max(lons.values),min(lats.values),max(lats.values)))

  iplat = lats.where( (lats >= latS ) & (lats <= latN), drop=True)
  iplon = lons.where( (lons >= lonW ) & (lons <= lonE), drop=True)
  # -- extract data in target region
  df0 = anm.sel(latitude=iplat,longitude=iplon)
  if season == "Monthly": 
    df = df0  
  else:
    # == seasonal mean
    if season == "ANN":
      df = annual_mean(df0) 
    else:
      df1 = season_mean(df0)
      if season == "DJF":
        df = df1[0::4,:,:]
      elif season == "MAM":
        df = df1[1::4,:,:]
      elif season == "JJA":
        df = df1[2::4,:,:]
      elif season == "SON":
        df = df1[3::4,:,:]
      del(df1)  
      #df = season_mean2(df0,season)

  del(iplat,iplon,clm,anm,da,df0)

  return df

def season_mean(ds, calendar="standard"):
    # Make a DataArray with the number of days in each month, size = len(time)
    month_length = ds.time.dt.days_in_month
    # Calculate the weights by grouping by 'time.season'
    weights = (month_length.groupby("time.season") / month_length.groupby("time.season").sum())
    # Test that the sum of the weights for each season is 1.0
    np.testing.assert_allclose(weights.groupby("time.season").sum().values, np.ones(4))
    # Setup our masking for nan values
    cond = ds.isnull()
    ones = xr.where(cond, 0.0, 1.0)
    # Calculate the numerator
    ds_sum = (ds * weights).resample(time="QS-DEC").sum(dim="time")
    # Calculate the denominator
    ones_out = (ones * weights).resample(time="QS-DEC").sum(dim="time")
    # Return the weighted average
    return ds_sum / ones_out
    #return ((ds * month_length).resample(time='QS-DEC').sum() / 
    #        month_length.resample(time='QS-DEC').sum())

def season_mean2(ds,season,calendar="standard"):
    anmS = ds.rolling(time=3, center=True).mean('time')
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
    return df 

def annual_mean(ds,calendar="standard"):
    """
    weight by days in each month
    """
    # Determine the month length
    month_length = ds.time.dt.days_in_month
    # Calculate the weights
    weights = (month_length.groupby("time.year") / month_length.groupby("time.year").sum())
    # Make sure the weights in each year add up to 1
    np.testing.assert_allclose(weights.groupby("time.year").sum(xr.ALL_DIMS), 1.0)
    # Setup our masking for nan values
    cond = ds.isnull()
    ones = xr.where(cond, 0.0, 1.0)
    # Calculate the numerator
    ds_sum = (ds * weights).resample(time="YS").sum(dim="time")
    # Calculate the denominator
    ones_out = (ones * weights).resample(time="YS").sum(dim="time")
    # Return the weighted average
    return ds_sum / ones_out
    #return (ds * weights).groupby("time.year").sum(dim="time")

def eof_analysis(da,season,regnam,mip,exp,relm,case,case_id,period,out_path,fig_path):
  '''
  perform eof analysis on selected quantity 
  '''
  var = da.attrs['name']
  vstr = da.attrs['name'].upper()
  vunt = da.attrs['units']

  #years = list(da.time.dt.year.data)
  times = da.time.dt.strftime("%Y-%m-%d")
  time_str = da.time.dt.strftime("%Y-%m-%d")
  ntime = len(times)
  print("number of total months in data: ", ntime)
  neof = 15

  lons,lats = da.longitude, da.latitude

  # -- EOF --
  for k in range(2):
    if k == 0: 
      l_detrend_eof = False 
      key = "raw"
      da_detrend = da
    else:
      l_detrend_eof = True  
      key = "detrend"
      da_detrend = detrend_dim(da,'time',1)
    da_detrend = da_detrend.sortby("latitude", ascending=True)
    da_detrend.attrs = da.attrs 
    da_detrend.attrs['name'] = var 
    da_detrend.attrs['units'] = vunt 

    clat = da_detrend['latitude'].astype(np.float64)
    clat = np.sqrt(np.cos(np.deg2rad(clat)))
  
    wanm = da_detrend.copy()
    wanm = da_detrend * clat
    wanm.attrs = da_detrend.attrs
    wanm.attrs['long_name'] = 'Wgt: {}({})'.format(vstr,vunt)
    xw_anm = wanm.transpose('time', 'latitude', 'longitude')
  
    #print("data min/max (raw): {} {}".format(da_detrend.min(),da_detrend.max()))
    #print("data min/max (weighted): {} {}".format(wanm.min(),wanm.max()))
    #print("data min/max (transpose): {} {}".format(xw_anm.min(),xw_anm.max()))

    eofs = eofunc_eofs(xw_anm.data, neofs=neof, meta=True)
  
    rpcs = eofunc_pcs(xw_anm.data, npcs=neof, meta=True)
    rpcs['time'] = da_detrend['time']
    rpcs.attrs['varianceFraction'] = eofs.attrs['varianceFraction']

    pcs = eofunc_pcs(xw_anm.data, npcs=neof, meta=True)
    pcs = pcs / pcs.std(dim='time')
    pcs['time'] = da_detrend['time']
    pcs.attrs['varianceFraction'] = eofs.attrs['varianceFraction']
    #print(pcs)
  
    do = xr.Dataset(coords = {
                    'time': da_detrend['time'],
                    'eof': np.arange(0,neof), 
                    'lat': xw_anm['latitude'], 
                    'lon': xw_anm['longitude']
                    },
                    attrs={'var_in': var,
                           'var_name': vstr,
                           'units': vunt},
                    )
    do['eofs']   = (['eof','lat','lon'],eofs.data)
    do['pcs']    = (['eof','time'],pcs.data)
    do['rawpcs'] = (['eof','time'],rpcs.data)
    do['varianceFraction'] = (['eof'],eofs.attrs['varianceFraction'].data)
    if not os.path.exists(out_path):
      os.makedirs(out_path)
    fout_name = "{}.{}.{}.{}.{}.{}.eof.{}.{}.{}.{}.nc".format(
                 mip,exp,case,relm,case_id,vstr,key,season,regnam,period)
    do.to_netcdf(os.path.join(out_path,fout_name)) 
    do.close()

    # Show the plot
    evec = xr.DataArray(data=eofs, dims=('eof','latitude','longitude'),
                        coords = {
                            'eof': np.arange(0,neof),
                            'lat': xw_anm['latitude'],
                            'lon': xw_anm['longitude']
                            })
    vmin = evec.min().values
    vmax = evec.max().values
    vmax = round(abs(max([vmin,vmax])),ndigits=2)
    vmin = -vmax 
    nlev = 21 

    axs = []
    fills = []
    neofsub = 5
    font = 12
    fig = plt.figure(figsize=(20, 8))
    grid = fig.add_gridspec(ncols=neofsub,nrows=3,hspace=0.4)  
    for ieof in range(neofsub): 
      ax1, fill1 = make_map_plot(evec[ieof,:,:],ieof, grid[0:2,ieof],vmin,vmax,nlev,font,fig,season)
      axs.append(ax1)
      fills.append(fill1)
      del(fill1,ax1)

    cbar = fig.colorbar(fills[neofsub-1],
                 ax=axs,
                 drawedges=True,
                 orientation='horizontal',
                 ticks = [vmin + (vmax-vmin)*i/(nlev-1) for i in range(nlev)], 
                 label='Eigenvector',
                 shrink=0.95,
                 aspect=80,
                 pad=0.05,
                 extendfrac='auto',
                 extendrect=True)
    cbar.ax.tick_params(labelsize=font*0.9)

    for ieof in range(neofsub):
      axs[ieof] = make_bar_plot(rpcs[ieof,:],ieof,grid[2,ieof],font,fig,vstr,vunt,season)
  
    fig.suptitle('EOF for {} ({})'.format(vstr,season), fontsize=font, y=0.9)

    if not os.path.exists(fig_path):
      os.makedirs(fig_path)
    fig_name = "fig_evec_pcs_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.pdf".format(
                mip,exp,case,relm,case_id,var,key,season,regnam,period)
    plt.savefig(os.path.join(fig_path, fig_name))
    plt.close()
    del(clat,wanm,xw_anm,eofs,do)
  return 

def make_map_plot(dat,ieof,grid_space,vmin,vmax,nlev,font,fig,season):
  lons,lats = dat.lon, dat.lat
  ny,nx = dat.shape
  x,y = np.meshgrid(lons,lats)

  # Fix the artifact of not-shown-data around 0 and 360-degree longitudes
  proj = ccrs.SouthPolarStereo(central_longitude=0.0,true_scale_latitude=None, globe=None)
  
  ax = fig.add_subplot(grid_space,projection=proj)
  ax.coastlines(linewidth=0.5, alpha=0.6)
  ax.set_extent([min(lons),max(lons),min(lats),max(lats)], ccrs.PlateCarree())
  ax.coastlines(linewidth=0.5, alpha=0.6)
  #ax.set_global()

  #plot grid lines
  gl = ax.gridlines(crs=ccrs.PlateCarree(),
                  draw_labels=True,
                  dms=False,
                  x_inline=False,
                  y_inline=False,
                  linewidth=1,
                  linestyle='dotted',
                  color="black",
                  alpha=0.3)

  # Manipulate latitude and longitude gridline numbers and spacing
  gl.ylocator = mticker.FixedLocator(np.arange(-90, -30, 15))
  gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, 30))
  
  # Manipulate longitude labels (0, 30 E, 60 E, ..., 30 W, etc.)
  ticks = np.arange(0, 210, 30)
  etick = ['0'] + [r'%dE' % tick for tick in ticks if (tick != 0) & (tick != 180)] + ['180']
  wtick = [r'%dW' % tick for tick in ticks[::-1] if (tick != 0) & (tick != 180)]
  labels = etick + wtick
  xticks = np.arange(0, 360, 30)
  yticks = np.full_like(xticks, -50)  # Latitude where the labels will be drawn
  
  for xtick, ytick, label in zip(xticks, yticks, labels):
    if label == '180':
        ax.text(xtick,
                ytick,
                label,
                fontsize=font*0.9,
                horizontalalignment='center',
                verticalalignment='top',
                transform=ccrs.Geodetic())
    elif label == '0':
        ax.text(xtick,
                ytick,
                label,
                fontsize=font*0.9,
                horizontalalignment='center',
                verticalalignment='bottom',
                transform=ccrs.Geodetic())
    else:
        ax.text(xtick,
                ytick,
                label,
                fontsize=font*0.9,
                horizontalalignment='center',
                verticalalignment='center',
                transform=ccrs.Geodetic())

  gl.bottom_labels = False
  gl.left_labels = False

  gl.top_labels = False
  gl.right_labels = False
  gl.rotate_labels = False

  # Set map boundary to include latitudes between 0 and 40 and
  # longitudes between -180 and 180 only
  #gv.set_map_boundary(ax, [-180, 180], [-90, -45], south_pad=1)
  
  # Import the default color map
  # newcmp = gvcmaps.BlueYellowRed
  #index  = [5, 20,  35, 50, 65, 85, 95, 110, 125,  0, 0, 135, 150,  165, 180, 200, 210, 220, 235, 250 ]
  # color_list = [newcmp[i].colors for i in index]
  # #-- Change to white
  # color_list[9]=[ 1., 1., 1.]
  # color_list[10]=[ 1., 1., 1.]
  newcmp = gvcmaps.BlueYellowRed
  color_list = newcmp(np.linspace(0, 1, nlev+1))
  #nmid = int((nlev+1)/ 2) 
  #color_list[nmid-1]=[ 1., 1., 1., 1.]
  #color_list[nmid]=[ 1., 1., 1., 1.]

  #-- Change to white
  # Define dictionary for kwargs
  levels = np.linspace(vmin,vmax,nlev+2)
  levels[1:nlev+1] = np.linspace(vmin,vmax,nlev)
  levels[0] = levels[0] - (vmax - vmin)/(nlev-1)
  levels[nlev+1] = levels[nlev] + (vmax - vmin)/(nlev-1)
  kwargs = dict(
    vmin = vmin,
    vmax = vmax,
    levels = levels, 
    colors=color_list,
    #add_colorbar=False,  # allow for colorbar specification later
    transform=ccrs.PlateCarree(),  # ds projection
  )

  # Contouf-plot U data (for filled contours)
  fillplot = ax.contourf(x,y,dat, **kwargs)

  # Draw map features on top of filled contour
  #ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=1)
  #ax.add_feature(cfeature.COASTLINE, edgecolor='gray', linewidth=0.5, zorder=1)
  ax.add_feature(cfeature.LAND, facecolor='none', zorder=1)
  ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=0.5, zorder=1)

  # Use geocat.viz.util convenience function to add titles to left and right of the plot axis.
  gvutil.set_titles_and_labels(ax,
                             lefttitle='EOF{} pattern({})'.format(ieof+1,season),
                             lefttitlefontsize=font*0.9,
                             righttitle='',
                             righttitlefontsize=font*0.9,
                             maintitle='',
                             xlabel="",
                             ylabel="")

  return ax, fillplot


def make_bar_plot(dataset,ieof,grid_space,font,fig,vstr,vunt,season):
    years  = int(dataset.time.dt.year.min())
    yeare  = int(dataset.time.dt.year.max())
    values = list(dataset.values)
    colors = ['blue' if val < 0 else 'red' for val in values]

    time  = dataset['time']
    xtime = np.linspace(1,len(time),len(time))
    if season == "Monthly":
      xtick = np.arange(0,len(time))
      xlabs = np.arange(0,len(time)) / 12.0  + years
      nint  = 240 
      ylabel = "Index"
      xlabel = "Time (year/month)"
    else:
      xtick = np.arange(0,len(time))
      xlabs = np.arange(0,len(time)) + years  
      nint  = 10
      ylabel = "Index"
      xlabel = "Time (year)"

    ax = fig.add_subplot(grid_space)

    ax.bar(xtime,
           values,
           color=colors,
           width=1.0,
           edgecolor='black',
           linewidth=0.1)

    # Use geocat.viz.util convenience function to add minor and major tick lines
    gvutil.add_major_minor_ticks(ax,
                                 x_minor_per_major=5,
                                 y_minor_per_major=5,
                                 labelsize=font*0.8)
    
    # Use geocat.viz.util convenience function to set axes tick values
    gvutil.set_axes_limits_and_ticks(ax,
                                     xlim=[min(xtick), max(xtick)],
                                     ylim=[min(dataset),max(dataset)])

    pct = dataset.attrs['varianceFraction'].values[ieof] * 100

    gvutil.set_titles_and_labels(ax,
                             lefttitle='PC{} ({},{})'.format(ieof+1,vstr,vunt),
                             lefttitlefontsize=font*0.9,
                             righttitle='{:.1f}%'.format(pct),
                             righttitlefontsize=font*0.9,
                             xlabel=xlabel,
                             ylabel=ylabel,
                             labelfontsize=font*0.8)

    #ax.set_xlim(0,len(dataset.time))
    ax.set_xticks(xtick[::nint])
    ax.set_xticklabels(xlabs[::nint].astype(int))
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()

    return ax

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
  out_path = os.path.join("/lcrc/group/e3sm/public_html/diagnostic_output/ac.szhang/polar_analysis/data","eof_analysis","raw_index")
  fig_path = os.path.join("/lcrc/group/e3sm/public_html/diagnostic_output/ac.szhang/polar_analysis/figure", "eof_analysis","figure")

  # region of interest (eof sector)
  eof_region = {'socn':{'west':0., 'east':360., 'south':-90., 'north':-40.}}
  mip       = "cmip6"
  exp       = "historical"
  relm      = "r1i1p1f1"
  case_id   = 'eof_gwmsl'
  tableId   = "Amon"
  varlist   = ["ts", "tas"]
  varstrs   = ["TS", "TREFHT"]
  period    = "195001-201412"
  pclimo    = "195001-198012"
  season    = "ANN" #"DJF" #"Monthly"
  run_path  = "/lcrc/group/e3sm/ac.szhang/acme_scratch/data/pcmdi/model/monthly"
  run_mask  = "/lcrc/group/e3sm/ac.szhang/acme_scratch/data/pcmdi/model/fixed/sftlf"
  for k,regnam in enumerate(eof_region.keys()):
    region = eof_region[regnam]
    for j,var in enumerate(varlist):
      ptmp = os.path.join(run_path,mip,exp,tableId,var)
      flst = sorted(glob.glob(os.path.join(ptmp,mip+"."+exp+".*."+relm+"*"+var+"*.nc")))
      for filePath in flst:
        if os.path.isfile(filePath):
          vstr = varstrs[j]
          fileName = filePath.split("/")[-1]
          case = fileName.split(".")[2]
          data_fil = filePath
          mask_fil = os.path.join(run_mask,tableId,mip+"."+exp+"."+case+".fx.sftlf.nc")
          clim_tmp = '{}.{}.{}.{}.climo.{}.{}.{}.{}.nc'.format(mip,exp,case,relm,case_id,vstr,regnam,pclimo)
          clim_fil = os.path.join(out_path,clim_tmp)
          case_dict = collections.OrderedDict()
          case_dict[case]   = Case(data_fil, var= vstr,    color="blue", label=case)
          case_dict['mask'] = Case(mask_fil, var= "sftlf", color="blue", label=case)
          case_dict['clim'] = Case(clim_fil, var= vstr,    color="blue", label=case)
          #call fuction to generate rgmn index 
          main(fig_path,out_path,mip,exp,relm,case_id,period,pclimo,case_dict,region,regnam,season)
