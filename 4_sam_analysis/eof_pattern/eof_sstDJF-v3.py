import numpy as np
import pandas as pd
import xarray as xr
import os
from datetime import datetime

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

from scipy import stats

import geocat.viz.util as gvutil
from geocat.viz import cmaps as gvcmaps
from geocat.comp import eofunc_eofs, eofunc_pcs


# ----- Parameter setting ------
ystr = 1982
yend = 2020

latS = -30.
latN = 30.
lonW = 120.
lonE = 290

neof = 3

# == Figure name ==
fnFIG = os.path.splitext(os.path.basename(__file__))[0]

# == netcdf file name and location"
fnc = 'oisst_monthly.nc'
dmask = xr.open_dataset('lsmask.nc')
print(dmask)

ds = xr.open_dataset(fnc)
print(ds)

# === Climatology and Anomalies
sst = ds.sst.where(dmask.mask.isel(time=0) == 1)
clm = sst.sel(time=slice(f'{ystr}-01-01',f'{yend}-12-01')).groupby('time.month').mean(dim='time')
anm = (sst.groupby('time.month') - clm)

# == seasonal mean
anmS = anm.rolling(time=3, center=True).mean('time')
anmDJF=anmS.sel(time=slice(f'{ystr}-02-01',f'{yend}-12-01',12))
print(anmDJF)

# -- Detorending
def detrend_dim(da, dim, deg=1):
    # detrend along a single dimension
    p = da.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    return da - fit

# -- EOF --
anmDJF=detrend_dim(anmDJF,'time',1)
anmDJF = anmDJF.sortby("lat", ascending=True)
clat = anmDJF['lat'].astype(np.float64)
clat = np.sqrt(np.cos(np.deg2rad(clat)))
wanm = anmDJF 
wanm = anmDJF * clat
wanm.attrs = anmDJF.attrs
wanm.attrs['long_name'] = 'Wgt: ' + wanm.attrs['long_name']
print(wanm)

lon = wanm['lon']
if ( ((lonW < 0) or (lonE < 0 )) and (lon.values.min() > -1) ):
   wanm=wanm.assign_coords(lon=( (lon + 180) % 360 - 180) )
   wanm = wanm.sortby("lon")
   print(' change longitude ')
 
print(wanm)  

xw = wanm.sel(lat=slice(latS, latN), lon=slice(lonW, lonE))
xw_anm = xw.transpose('time', 'lat', 'lon')
print(xw_anm)

eofs = eofunc_eofs(xw_anm.data, neofs=neof, meta=True)
pcs = eofunc_pcs(xw_anm.data, npcs=neof, meta=True)
pcs = pcs / pcs.std(dim='time')
pcs['time']=anmDJF['time']
pcs.attrs['varianceFraction'] = eofs.attrs['varianceFraction']
print(pcs)

evec = xr.DataArray(data=eofs, dims=('eof','lat','lon'),
    coords = {'eof': np.arange(0,neof), 'lat': xw['lat'], 'lon': xw['lon']} )
print(evec)

# ------

# Correlation & Regression
cor1 = xr.corr(pcs[0,:], anmDJF, dim="time")
cor2 = xr.corr(pcs[1,:], anmDJF, dim="time")
cor3 = xr.corr(pcs[2,:], anmDJF, dim="time")
reg1 = xr.cov(pcs[0,:], anmDJF, dim="time")/pcs[0,:].var(dim='time',skipna=True).values
reg2 = xr.cov(pcs[1,:], anmDJF, dim="time")/pcs[1,:].var(dim='time',skipna=True).values
reg3 = xr.cov(pcs[2,:], anmDJF, dim="time")/pcs[2,:].var(dim='time',skipna=True).values


# -- figure plot

def makefig(dat, ieof, grid_space):
  # Fix the artifact of not-shown-data around 0 and 360-degree longitudes
  # Generate axes using Cartopy to draw coastlines
  ax = fig.add_subplot(grid_space,
          projection=ccrs.PlateCarree(central_longitude=180))
#          projection=ccrs.Robinson(central_longitude=210))
  ax.coastlines(linewidth=0.5, alpha=0.6)
  
  gl = ax.gridlines(crs=ccrs.PlateCarree(),
                  draw_labels=True,
                  dms=False,
                  x_inline=False,
                  y_inline=False,
                  linewidth=1,
                  linestyle='dotted',
                  color="black",
                  alpha=0.3)
  gl.top_labels = False
  gl.right_labels = False
  gl.rotate_labels = False

  # Use geocat.viz.util convenience function to add minor and major tick lines
  gvutil.add_major_minor_ticks(ax, labelsize=10)
  
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
    vmin = -0.06,
    vmax = 0.06,
    levels = 21,
    colors=color_list,
    add_colorbar=False,  # allow for colorbar specification later
    transform=ccrs.PlateCarree(),  # ds projection
  )
  
  # Contouf-plot U data (for filled contours)
  fillplot = dat[ieof,:,:].plot.contourf(ax=ax,  **kwargs)

  # Draw map features on top of filled contour
  ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=1)
  ax.add_feature(cfeature.COASTLINE, edgecolor='gray', linewidth=0.5, zorder=1)
  
  # Use geocat.viz.util convenience function to add titles to left and right of the plot axis.
  gvutil.set_titles_and_labels(ax,
                             lefttitle=f'EOF{ieof+1} pattern',
                             lefttitlefontsize=12,
                             righttitle='',
                             righttitlefontsize=12,
                             maintitle='',
                             xlabel="",
                             ylabel="")

  return ax, fillplot




def make_bar_plot(dataset, ieof, grid_space):
    years = list(dataset.time.dt.year)
    values = list(dataset[ieof,:].values)
    colors = ['blue' if val < 0 else 'red' for val in values]

    ax = fig.add_subplot(grid_space)

    ax.bar(years,
           values,
           color=colors,
           width=1.0,
           edgecolor='black',
           linewidth=0.5)

    # Use geocat.viz.util convenience function to add minor and major tick lines
    gvutil.add_major_minor_ticks(ax,
                                 x_minor_per_major=5,
                                 y_minor_per_major=5,
                                 labelsize=10)

    # Use geocat.viz.util convenience function to set axes tick values
    gvutil.set_axes_limits_and_ticks(ax,
                                     xticks=np.linspace(1980, 2020, 5),
                                     xlim=[1979.5, 2020.5],
                                     ylim=[-3.0, 3.5])

    pct = dataset.attrs['varianceFraction'].values[ieof] * 100
    print(pct)
    gvutil.set_titles_and_labels(ax,
                             lefttitle=f'PC{ieof+1} (normalized)',
                             lefttitlefontsize=12,
                             righttitle=f'{pct:.1f}%',
                             righttitlefontsize=12,
                             xlabel="Year",
                             ylabel="",
                             labelfontsize=10 )
    return ax


# Show the plot

fig = plt.figure(figsize=(14, 6))
grid = fig.add_gridspec(ncols=3, nrows=3, hspace=0.4)

ax1, fill1 = makefig(evec,0, grid[0:2,0])
ax2, fill2 = makefig(evec,1, grid[0:2,1])
ax3, fill3 = makefig(evec,2, grid[0:2,2])

fig.colorbar(fill2,
                 ax=[ax1,ax2,ax3],
                 ticks=np.linspace(-0.06, 0.06, 5),
                 drawedges=True,
                 label='Eigenvector',
                 orientation='horizontal',
                 shrink=0.3,
                 pad=0.08,
                 extendfrac='auto',
                 extendrect=True)

ax1 = make_bar_plot(pcs,0,grid[2,0])
ax2 = make_bar_plot(pcs,1,grid[2,1])
ax3 = make_bar_plot(pcs,2,grid[2,2])

 
fig.suptitle('EOF for SST (DJF)', fontsize=16, y=0.9)

plt.draw()

plt.savefig(fnFIG+".png")
#plt.savefig(fnFIG+".pdf")
#plt.savefig(fnFIG+".eps", format='eps')

