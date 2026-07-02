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


# ----- Parameter setting ------

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
clm = sst.sel(time=slice('1982-01-01','2020-12-01')).groupby('time.month').mean(dim='time')
anm = (sst.groupby('time.month') - clm)
#print(clm)

# -- regional average
def wgt_areaave(indat, latS, latN, lonW, lonE):
  lat=indat.lat
  lon=indat.lon

  if ( ((lonW < 0) or (lonE < 0 )) and (lon.values.min() > -1) ):
     anm=indat.assign_coords(lon=( (lon + 180) % 360 - 180) )
     lon=( (lon + 180) % 360 - 180)
  else:
     anm=indat

  iplat = lat.where( (lat >= latS ) & (lat <= latN), drop=True)
  iplon = lon.where( (lon >= lonW ) & (lon <= lonE), drop=True)

#  print(iplat)
#  print(iplon)
  wgt = np.cos(np.deg2rad(lat))
  odat=anm.sel(lat=iplat,lon=iplon).weighted(wgt).mean(("lon", "lat"), skipna=True)
  return(odat)

# -- Calculate nino3.4 index
nino=wgt_areaave(anm,-5,5,-170,-120)
ninoSD=nino/nino.std(dim='time')
rninoSD=ninoSD.rolling(time=7, center=True).mean('time')

# -- Detorending
def detrend_dim(da, dim, deg=1):
    # detrend along a single dimension
    p = da.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    return da - fit

# -- Running mean
ranm = anm.rolling(time=7, center=True).mean('time')
rdanm = detrend_dim(ranm,'time',1)


# -- Correlation & Regression
ranm = anm.rolling(time=7, center=True).mean('time')

# Leading
corM12 = xr.corr(rninoSD, rdanm.shift(time=-12), dim="time")
regM12 = xr.cov( rninoSD, rdanm.shift(time=-12), dim="time")/rninoSD.var(dim='time',skipna=True).values
corM6 = xr.corr(rninoSD, rdanm.shift(time=-6), dim="time")
regM6 = xr.cov( rninoSD, rdanm.shift(time=-6), dim="time")/rninoSD.var(dim='time',skipna=True).values

# simultaneous
cor0 = xr.corr(rninoSD, rdanm, dim="time")
reg0 = xr.cov(rninoSD, rdanm, dim="time")/rninoSD.var(dim='time',skipna=True).values

# Laging
corP6 = xr.corr(rninoSD, rdanm.shift(time=6), dim="time")
regP6 = xr.cov( rninoSD, rdanm.shift(time=6), dim="time")/rninoSD.var(dim='time',skipna=True).values
corP12 = xr.corr(rninoSD, rdanm.shift(time=12), dim="time")
regP12 = xr.cov( rninoSD, rdanm.shift(time=12), dim="time")/rninoSD.var(dim='time',skipna=True).values
corP18 = xr.corr(rninoSD, rdanm.shift(time=18), dim="time")
regP18 = xr.cov( rninoSD, rdanm.shift(time=18), dim="time")/rninoSD.var(dim='time',skipna=True).values


# -- figure plot

def makefig(cor, title, grid_space):
  # Fix the artifact of not-shown-data around 0 and 360-degree longitudes
  cor = gvutil.xr_add_cyclic_longitudes(cor, 'lon')
  # Generate axes using Cartopy to draw coastlines
  ax = fig.add_subplot(grid_space,
          projection=ccrs.Robinson(central_longitude=210))
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
    vmin = -1.0,
    vmax = 1.0,
    levels = 21,
    colors=color_list,
    add_colorbar=False,  # allow for colorbar specification later
    transform=ccrs.PlateCarree(),  # ds projection
  )
  
  # Contouf-plot U data (for filled contours)
  fillplot = cor.plot.contourf(ax=ax,  **kwargs)

  # Draw map features on top of filled contour
  ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=1)
  ax.add_feature(cfeature.COASTLINE, edgecolor='gray', linewidth=0.5, zorder=1)
  
  # Plot Hatch
 
  df = 40 
  sig=xr.DataArray(data=cor.values*np.sqrt((df-2)/(1-np.square(cor.values))),
      dims=["lat","lon'"],
      coords=[cor.lat, cor.lon])
  t90=stats.t.ppf(1-0.05, df-2)
  t95=stats.t.ppf(1-0.025, df-2)
  sig.plot.contourf(ax=ax,levels = [-1*t95, -1*t90, t90, t95], colors='none', 
      hatches=['..', None, None, None, '..'], extend='both', 
      add_colorbar=False, transform=ccrs.PlateCarree())
  
  # Use geocat.viz.util convenience function to add titles to left and right of the plot axis.
  gvutil.set_titles_and_labels(ax,
                             lefttitle=title,
                             lefttitlefontsize=16,
                             righttitle='',
                             righttitlefontsize=16,
                             xlabel="",
                             ylabel="")

  return ax, fillplot

# Show the plot

fig = plt.figure(figsize=(10, 12))
grid = fig.add_gridspec(ncols=2, nrows=3)
#grid = fig.add_gridspec(ncols=2, nrows=3, hspace=-0.20)


ax1, fill1 = makefig(corP18,'18-month lag', grid[0,0])
ax2, fill2 = makefig(corP12,'12-month lag', grid[1,0])
ax3, fill3 = makefig(corP6,'6-month lag', grid[2,0])
ax4, fill4 = makefig(cor0,'Simultaneous', grid[0,1])
ax5, fill5 = makefig(corM6,'6-month lead', grid[1,1])
ax6, fill6 = makefig(corM12,'12-month lead', grid[2,1])

fig.colorbar(fill6,
                 ax=[ax1, ax2, ax3, ax4, ax5, ax6],
#                 ticks=np.linspace(-5, 5, 11),
                 drawedges=True,
                 orientation='horizontal',
                 shrink=0.5,
                 pad=0.05,
                 extendfrac='auto',
                 extendrect=True)
  
fig.suptitle('SST correlation with Nino3.4 (>95%)', fontsize=18, y=0.9)

plt.draw()

plt.savefig(fnFIG+".png")
#plt.savefig(fnFIG+".pdf")
#plt.savefig(fnFIG+".eps", format='eps')

