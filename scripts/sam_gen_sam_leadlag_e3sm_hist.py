import os
import collections
import xarray as xr
import xskillscore as xs
import numpy as np
import pandas as pd
import glob

from datetime import datetime
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from global_land_mask import globe

from scipy.stats import pearsonr
from scipy.signal import butter, filtfilt, sosfilt,lfilter

from mpl_toolkits.basemap import Basemap

from matplotlib.pylab import rcParams
from matplotlib.patches import Polygon

import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

import cmaps as gvcmaps
import geocat.viz.util as gvutil
import geocat.viz as gv

def main(fig_path,out_path,mip,exp,relm,case_id,period,case_dict,sam_region,l_check_sam_region):
  for key in case_dict:
    if key == "mask":
      dmsk = case_dict[key].path
      vmsk = case_dict[key]._var
    elif key == "sam":
      dsam = case_dict[key].path
      vsam = case_dict[key]._var
    else:
      case = key 
      var  = case_dict[key]._var
      data = case_dict[key].path
  
  print("working on ",case,var)

  #load model data 
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

  #load index file
  if os.path.exists(dsam):
    sam_df = pd.read_csv(dsam,index_col=False)
  else: 
    print(dsam)
    exit("sam index file not found ....")
  #perform lead-lag analysis
  time = sam_df['time']
  sam_df = sam_df.set_index('time')
  l_sam_exist = False
  for col in sam_df.columns: 
    if 'sam_idx' in col: 
      sam = sam_df[col].to_xarray() 
      l_sam_exist = True 

  if not l_sam_exist: 
    exit('sam index not found in {}'.format(dsam))
  else: 
    sam = sam.assign_coords({"time": da.time})
  
  #finally perform the lead lag analysis 
  sam_lead_lag_anl(mip,exp,relm,case,case_id,out_path,fig_path,da,sam,time) 

  return

# -- Detrending
def detrend_dim(da, dim, deg=1):
    # detrend along a single dimension
    p = da.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    return da - fit

def sam_lead_lag_anl(mip,exp,relm,case,case_id,out_path,fig_path,da,sam,time):
  lons    = da['longitude'][:]
  lats    = da['latitude'][:]
  clm     = da.groupby('time.month').mean(dim='time')
  anm     = (da.groupby('time.month') - clm)
  vstr    = da.name 
  vunt    = da.units
  samSD  = sam/sam.std(dim='time')

  # -- filtering (or Running mean)
  #rsamSD    = samSD.rolling(time=5, center=True).mean('time')
  #ranm       = anm.rolling(time=5, center=True).mean('time')
  rsamSD     = samSD
  ranm        = anm
  rsamSD[:]  = low_pass(1.0/5.0,samSD,axis=0) #5-month: 1/5.0
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

  # Lead/Lag regression and correlation  
  leadlag = np.arange(-12,18,3)
  leadlag = leadlag[::-1]
  corll = [] 
  regll = []
  pvlll = []
  for tllg in leadlag: 
    #negative(minus): leading, positive(plus): laging, zero: simutaneous
    cor = xr.corr(rsamSD, rdanm.shift(time=tllg), dim="time")
    reg = xr.cov(rsamSD, rdanm.shift(time=tllg), dim="time")/rsamSD.var(dim='time',skipna=True).values
    pvl = xs.pearson_r_p_value(rsamSD, rdanm.shift(time=tllg), dim="time",skipna=True)
    corll.append(cor)
    regll.append(reg)
    pvlll.append(pvl)
    del(cor,reg,pvl)

  #save data to nc file 
  save_figure_data(corll,regll,pvlll,leadlag,mip,exp,relm,case,case_id,period,var,vunt,out_path)

  # Show the plot
  #vmin  = np.nanmin(np.array(regll))
  #vmax  = np.nanmax(np.array(regll))
  #vmax  = max([abs(vmin),abs(vmax)])
  #vmin  = vma * -1.0
  vmax  =  1.0
  vmin  = -1.0
  nlev  =  21
  ncol  = 2
  nrow  = int(len(leadlag)/ncol)
  fontsize = 16
  bartitle = 'Regressed {}({})'.format(vstr,vunt)
  #fig = plt.figure(figsize=(12, 24))
  fig = plt.figure(figsize=(16, 30))
 #grid = fig.add_gridspec(ncols=2, nrows=3, hspace=-0.20)
  grid = fig.add_gridspec(ncols=ncol, nrows=nrow)
  axs = [] 
  fills = []
  for j in range(ncol):
    for i in range(nrow):
      k = i + j*nrow #len(leadlag) - j*ncol - i - 1
      tllg = leadlag[k]
      if tllg > 0.0: 
        tag = '{}-month {}'.format(abs(tllg),'lag')
      elif tllg < 0.0:
        tag = '{}-month {}'.format(abs(tllg),'lead')
      else: 
        tag = 'Simultaneous' 
      print(i,j,k,tag)
      ax, fill = draw_regression_map(vstr,vunt,lons,lats,corll[k],regll[k],pvlll[k],fig,sam_region,
                                     fontsize,tag,grid[i,j],vmin,vmax,nlev)
      axs.append(ax)
      fills.append(fill)
      del(ax,fill)

  cb = fig.colorbar(fills[len(leadlag)-1],
                  ax=axs,
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

  fig.suptitle('Corr. & Regress with SAM: {}({})'.format(vstr,vunt), fontsize=fontsize, y=0.9)
  plt.rcParams["font.family"] = "sans-serif"
  plt.draw()

  if not os.path.exists(fig_path):
    os.makedirs(fig_path)
  fig_name = "fig_regress_leadlag_map_{}_{}_{}_{}_{}_{}_{}.pdf".format(mip,exp,case,relm,case_id,var,period)
  plt.savefig(os.path.join(fig_path,fig_name))
  plt.close()

  return 

def save_figure_data(vcor,vreg,pval,leadlag,mip,exp,relm,case,case_id,period,var,vunt,out_path):
  #create xarray to save data
  lons,lats = vcor[0].longitude,vcor[0].latitude
  df = xr.Dataset(
          { "var_reg": (["leadlag","lat", "lon"], np.array(vreg)),
            "var_cor": (["leadlag","lat", "lon"], np.array(vcor)),
            "pval": (["leadlag","lat", "lon"],  np.array(pval)),
            },
          coords={
              "leadlag": (["leadlag"], np.array(leadlag)),
              "lon": (["lon"], lons.data),
              "lat": (["lat"], lats.data),
              },
          attrs=dict(description="sam lead/lag regression maps",
                    reference_time=period),
          )

  df.leadlag.attrs["units"] = "months"
  df.lat.attrs["units"] = "degree_north"
  df.lon.attrs["units"] = "degree_east"
  df.var_cor.attrs["units"] = "1"
  df.var_cor.attrs["long_name"] = var + " correlation"
  df.var_reg.attrs["units"] = vunt
  df.var_reg.attrs["long_name"] = var + " regression"
  df.pval.attrs["units"] = "1"
  df.pval.attrs["long_name"] = "p values"

  #save data to netcdf 
  if not os.path.exists(out_path):
    os.makedirs(out_path)
  fout_name = "{}_{}_{}_{}_{}_{}_leadlag_{}.nc".format(mip,exp,case,relm,case_id,var,period)
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
  #ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=1)
  #ax.add_feature(cfeature.COASTLINE, edgecolor='gray', linewidth=0.5, zorder=1)
  ax.add_feature(cfeature.LAND, facecolor='none', zorder=1)
  ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=0.5, zorder=1)


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
  #draw_regional_box(region)
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

def low_pass(cutoff_freq, data, order=5, axis=-1):
    #low-pass: filtering high-frequency signal
    #nyquist normalized cutoff for digital design
    Wn = cutoff_freq 
    b, a = butter(order, Wn, btype='lowpass', analog=False)
    data_filt = filtfilt(b, a, data, axis=axis, method="gust")
    return data_filt

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

def define_sam(da,region):
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
 
  sam_sst    = wgt_areaave(da,region) 
  sam_clm    = wgt_areaave(dclm,region)
  sam_ind    = wgt_areaave(anm,region) 
  samSD_ind  = sam_ind / sam_ind.std(dim='time')
  #instead of moving average, using a low-pass filtering 
  #rsam_ind   = sam_ind.rolling(time=5, center=True).mean('time')
  #osam_ind   = sam_ind.rolling(time=3, center=True).mean('time')
  #rsamSD_ind = samSD_ind.rolling(time=5, center=True).mean('time')
  #osamSD_ind = samSD_ind.rolling(time=3, center=True).mean('time')
  rsam_ind    = low_pass(1.0/5.0,sam_ind,axis=0)   #5-month: 1/5.0
  osam_ind    = low_pass(1.0/3.0,sam_ind,axis=0)   #3-month: 1/3.0
  rsamSD_ind  = low_pass(1.0/5.0,samSD_ind,axis=0) #5-month: 1/5.0
  osamSD_ind  = low_pass(1.0/3.0,samSD_ind,axis=0) #3-month: 1/3.0

  df = pd.DataFrame()
  df['samsst_val(degC)'] = sam_sst
  df['samsst_clm(degC)'] = sam_clm
  df['sam_idx(degC)']    = sam_ind
  df['samSD_idx(1)']     = samSD_ind
  df['rsam_idx(degC)']   = rsam_ind
  df['rsamSD_idx(1)']    = rsamSD_ind
  df['osam_idx(degC)']   = osam_ind
  df['osamSD_idx(1)']    = osamSD_ind
  df['time']              = time_str
  
  ### re-order columns
  df = df[['time','samsst_val(degC)','samsst_clm(degC)','sam_idx(degC)','rsam_idx(degC)','osam_idx(degC)','samSD_idx(1)','rsamSD_idx(1)','osamSD_idx(1)']]
  ### clean-up DataFrame
  df = df.reset_index(drop=True)
  del(clm,anm,dclm,da,sam_sst,sam_clm,sam_ind,rsam_ind,samSD_ind,rsamSD_ind,times,time_str)
  return df

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
  out_path = os.path.join("/lcrc/group/e3sm/public_html/diagnostic_output/ac.szhang/polar_analysis/data","sam_analysis","raw_index")
  fig_path = os.path.join("/lcrc/group/e3sm/public_html/diagnostic_output/ac.szhang/polar_analysis/figure", "sam_analysis","lead_lag_analysis","figure")

  # region of interest (sam sector)
  sam_region   = {'west':-180., 'east':180., 'south':-90., 'north':-20.}

  #sanity check (plot region with and without mask) 
  l_check_sam_region = False #True  

  mip       = "e3sm"
  exps      = [ "historical"]
  relms     = [ "0701", "0751", "0801" ]
  product   = "v2_1-SORRM"
  tableId   = "Amon"
  case_id   = 'sam_gwmsl'
  varlist   = ['ts','tas',   'psl','pr',   'tauu','tauv']
  varstrs   = ['TS','TREFHT','PSL','PRECT','TAUX','TAUY']
  period    = "195001-201412"
  run_path  = "/lcrc/group/e3sm/ac.szhang/acme_scratch/data/pcmdi/model/monthly"
  run_mask  = "/lcrc/group/e3sm/ac.szhang/acme_scratch/data/pcmdi/model/fixed/sftlf"
  for k,var in enumerate(varlist):
    for exp in exps:
      for relm in relms:
        varstr = varstrs[k]
        case = product
        ptmp = os.path.join(run_path,mip,exp,tableId,var)
        ftmp = '{}.{}.{}.{}.*.{}.{}.nc'.format(mip,exp,product,relm,var,period)
        filePath = sorted(glob.glob(os.path.join(ptmp,ftmp)))
        if len(filePath) > 0 and os.path.isfile(filePath[0]):
          filePath = filePath[0]
          fileName = filePath.split("/")[-1]
          case = fileName.split(".")[2]
          data_fil = filePath
          mask_fil = os.path.join(run_mask,tableId,mip+"."+exp+"."+case+".fx.sftlf.nc")
          index_fil = os.path.join(out_path,'{}.{}.{}.{}.{}.{}.{}.csv'.format(mip,exp,case,relm,case_id,'PSL',period))
          print(data_fil)
          print(index_fil)
          case_dict = collections.OrderedDict()
          case_dict[case] = Case(data_fil, var= varstr.upper(), color="blue", label=case)
          case_dict['mask'] = Case(mask_fil, var="sftlf", color="blue", label=case)
          case_dict['sam'] = Case(index_fil, var = "sam", color="blue", label=case)
          #call fuction to generate regression 
          main(fig_path,out_path,mip,exp,relm,case_id,period,case_dict,sam_region,l_check_sam_region)
