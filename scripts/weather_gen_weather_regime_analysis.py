import os
import collections
import xarray as xr
import xskillscore as xs
import numpy as np
import pandas as pd
import glob
import time

from datetime import datetime
from skimage.feature import peak_local_max

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cartopy.crs as ccrs

from collections import OrderedDict
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

from sk_kmean_don import * 

def main(fig_path,out_path,mip,exp,relm,case_dict,case_id,region,regnam,period,season,use_sk_kmean_don,n_cluster):
  for key in case_dict:
    if key == "mask":
      dmsk = case_dict[key].path
      vmsk = case_dict[key]._var
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

  #construct 3-D anomaly time series for selected region 
  df = find_regional_data(da,region)
  df['name'] = da.name
  df['units'] = da.units

  #Perform KMeans clustering to idenfity weather regimes
  if use_sk_kmean_don: 
    mkclass = KMeans_Skdon(df,season,n_cluster,mip,exp,relm,case,case_id,region,regnam,period,out_path,fig_path)
  else:
    mkclass = KMeans_Clustering(df,season,n_cluster,mip,exp,relm,case,case_id,region,regnam,period,out_path,fig_path)
 
  return

def draw_weather_regime_map(da,mk,n_cluster,mip,exp,relm,case,case_id,region,regnam,period,fig_path):  
  latN = region['north']
  latS = region['south']
  lonE = region['east']
  lonW = region['west']

  var  = da.name
  vstr = da.name.upper()
  vunt = da.units.values

  vmin = -3.0
  vmax =  3.0 
  nlev = 11

  if n_cluster < 3:
    ratio = 1.0 / 4.0
  else: 
    ratio = int(n_cluster/4) / 4.0

  levels = np.linspace(vmin,vmax,nlev)

  nt,ny,nx = da.shape
  x,y = np.meshgrid(da.longitude, da.latitude)

  proj = ccrs.SouthPolarStereo(central_longitude=0.0,true_scale_latitude=None, globe=None)
  
  if n_cluster < 3: 
    fig, axes = plt.subplots(1,4,figsize=(8*n_cluster,8*n_cluster), subplot_kw=dict(projection=proj))
  else:
    fig, axes = plt.subplots(int(n_cluster/4+0.5),4,figsize=(8,8), subplot_kw=dict(projection=proj))

  #regimes = ['NAO$^-$', 'NAO$^+$', 'Blocking', 'Atlantic Ridge']
  #regimes = ['Regime 1','Regime 2','Regime 3','Regime 4','Regime 5','Regime 6']
  tags = list('abcdefghijklmn')
  regimes = []
  for i in range(n_cluster):
    regimes.append('Regime {}'.format(i+1))
    onecen = mk.cluster_centers_[i,:].reshape(ny,nx, order='F')
    cs = axes.flat[i].contourf(x, y, onecen,
                               levels=levels, 
                               transform=ccrs.PlateCarree(),
                               cmap='RdBu_r')
    cb=fig.colorbar(cs, ax=axes.flat[i], shrink=0.8, aspect=20)
    cb.set_label('{}({})'.format(vstr,vunt),labelpad=-7)
    axes.flat[i].set_extent([lonW,lonE,latS,latN], ccrs.PlateCarree())
    axes.flat[i].coastlines()
    axes.flat[i].gridlines()
    #axes.flat[i].set_global()

    title = '{}, {:4.1f}%'.format(regimes[i], get_cluster_fraction(mk, i)*100)
    axes.flat[i].set_title(title)
    plt.text(0, 1, '', #tags[i],
             transform=axes.flat[i].transAxes,
             va='bottom',
             fontsize=plt.rcParams['font.size']*2,
             fontweight='bold')

  if not os.path.exists(fig_path):
    os.makedirs(fig_path)
  fig_name = "fig_{}cluster_map_{}_{}_{}_{}_{}_{}_{}.pdf".format(n_cluster,regnam,mip,exp,case,relm,case_id,vstr,period)
  plt.savefig(os.path.join(fig_path, fig_name))
  plt.close()

  return

def find_regional_data(da,region):

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
  #lon_name = "longitude"
  #ds['_longitude_adjusted'] = xr.where(ds[lon_name] > 180,ds[lon_name] - 360,ds[lon_name])
  #ds = (ds.swap_dims({lon_name: '_longitude_adjusted'})
  #        .sel(**{'_longitude_adjusted': sorted(ds._longitude_adjusted)})
  #        .drop(lon_name))
  #ds = ds.rename({'_longitude_adjusted': lon_name})
  clm  = da.groupby('time.month').mean(dim='time')
  anm  = (da.groupby('time.month') - clm)

  iplat = lats.where( (lats >= latS ) & (lats <= latN), drop=True)
  iplon = lons.where( (lons >= lonW ) & (lons <= lonE), drop=True)

  # -- extract data in target region 
  df = anm.sel(latitude=iplat,longitude=iplon)

  del(iplat,iplon,clm,anm,da)

  return df

def get_cluster_fraction(m, label):
        return (m.labels_==label).sum()/(m.labels_.size*1.0)

def KMeans_Skdon(da,season,n_cluster,mip,exp,relm,case,case_id,region,regnam,period,out_path,fig_path):
  '''
  da for one point in time (with lats x lons)
  '''
  l_diag_figure = True 
  ###############################################
  #sim = “ssim”, “str”, “ed”, “md” 
  #ssim: mean structural similarity 
  #str: Pearson correlation coefficient
  #ed: Euclidean distance
  #md: Manhattan distance
  ##############################################
  sim = "ssim" 

  ##############################################
  #ini = “rand” or “pp” 
  #rand: randomized initialization
  #pp: k-means ++ initialization scheme
  ##############################################
  ini = "pp"
  
  #output directory 
  if not os.path.exists(out_path):
    os.makedirs(out_path)

  if season == "DJF":
    dsub = da.sel(time=da.time.dt.month.isin([1,2,12]), drop=True)
  elif season == "JJA":
    dsub = da.sel(time=da.time.dt.month.isin([6,7,8]), drop=True)
  elif season == "SON":
    dsub = da.sel(time=da.time.dt.month.isin([9,10,11]), drop=True)
  elif season == "MAM":
    dsub = da.sel(time=da.time.dt.month.isin([3,4,5]), drop=True)
  else:
    #use all avaiable data   
    dsub = da

  key = '{}_{}'.format(da.name.upper(),season)

  date = pd.to_datetime(datetime.now()).strftime('%Y%m%d_%H%M%S')

  shape = dsub.shape
  dims = dsub.dims
  coords = dsub.coords

  #extract and reorgnize data
  data = dsub.values
  nt,ny,nx = data.shape  
  data = np.reshape(data, [nt, ny*nx], order='F')

  #start to process the data 
  nt = 0
  times = []
  
  #directory to save skmean analysis data
  odir0 = os.path.join(out_path,key)
  if not os.path.exists(odir0):
    os.makedirs(odir0)

  # =================================
  # silhouette analysis
  # calculate similarity in-between
  # =================================
  ofile_sbtw = "{}_sim_btw_{}_{}_{}_{}_{}_{}_{}.nc".format(
                sim,mip,exp,case,relm,case_id,regnam,period)
  ofile_sbtw = os.path.join(out_path,ofile_sbtw)
  if not os.path.exists(ofile_sbtw):
    ss = selsim(data,sim)
    ny,nx = ss.shape  
    xx = np.arange(nx)
    yy = np.arange(ny)
    xx, yy = np.meshgrid(xx,yy)
    do = xr.Dataset(
            data_vars=dict(
                sim_btw=(["from", "to"], ss),
                ),
            coords=dict(
                x=("from", coords[dims[0]].data),
                y=("to", coords[dims[0]].data),
                ),
            attrs=dict(description="Weather related data."),
            )
    try: do[dims[0]] = coords[dims[0]]
    except: print('coords')
    encoding = {dims[0]:{'zlib': False,'_FillValue': -9999},
                'x':{'zlib': False,'_FillValue': -9999},
                'y':{'zlib': False,'_FillValue': -9999}, 
                'sim_btw':{'zlib': True,'_FillValue':-9999}}
    do.to_netcdf(ofile_sbtw,encoding=encoding)
    do.close() 
    if l_diag_figure:
      draw_silhouette_analysis_similarity(ss,sim,mip,exp,case,relm,case_id,regnam,period,fig_path)
      draw_silhouette_analysis_matrix(ss,sim,mip,exp,case,relm,case_id,regnam,period,fig_path)
    del(encoding,do,ss)
    
  for n in n_cluster: 
    odir = os.path.join(odir0,'{:02d}'.format(n),ini)
    ofile = "{}_{}_{}_{}_{}_{}_{}_{}.nc".format(
            sim,mip,exp,case,relm,case_id,regnam,period)
    ofile = os.path.join(odir,ofile)
    # =============================
    # run program for K means
    # =============================
    if not os.path.exists(odir): 
      os.makedirs(odir)  
    start_time = time.time()
    o = K_means(data,k=n,sim=sim,ini=ini)
    duration =  time.time() - start_time
    times.append([nt,key,sim,n,ini,duration])
    mkclass = o
    encoding = {}
    for iik,xx in enumerate(list(dims)+['centroids','cluster']):
      if xx not in encoding.keys():
        encoding[xx] = {'zlib': False,'_FillValue':-9999}
    print(ofile)
    do = xr.Dataset(coords = coords ) 
    C = o['C_fin']
    s1 = [n] + list(shape[1:])
    y = C.reshape( s1 )
    do['centroids'] = (['n']+ list(dims[1:]),  y)
    do['cluster'] = (dims[0],o['Clustering'])
    do.to_netcdf(ofile,encoding=encoding) 
    do.close()
    #print(o['Clustering'].shape)
    #print(o['Clustering'].min())
    #print(o['Clustering'].max())    
    del(do,o,C,s1,y,start_time,encoding,duration)

    # =============================
    # silhouette analysis
    # =============================
    print()
    do = xr.open_dataset(ofile)
    ds = xr.open_dataset(ofile_sbtw)
    ss = 1 - normalize1(ds['sim_btw'].values)
    clus = do['cluster']
    #print(clus.shape)
    #exit()
    df = Silh_s(data,ss,clus)
    fout = '{}_S-anal_{}_{}_{}_{}_{}_{}_{}.csv'.format(
            sim,mip,exp,case,relm,case_id,regnam,period)
    print(os.path.join(odir,fout))
    df.to_csv(os.path.join(odir,fout))
    if l_diag_figure:
      draw_clustering_analysis(sim,mkclass,data,df,mip,exp,case,relm,case_id,regnam,period,fig_path)
    do.close()
    ds.close()
    del(do,ds,ss,clus,df,fout)
 
  if 0: # write time record to file
    dt = pd.DataFrame(times, columns=['run', 'test', 'sim', 'size', 'ini', 'dur'])
    fout = 'runtime_{}_{}_{}_{}_{}_{}_{}.csv'.format(mip,exp,case,relm,case_id,regnam,period)
    dt.to_csv(outdir + 'runtime.csv', index=None) 
  
  if l_diag_figure:
    draw_clustering_uncertainty(fout,ss,mip,exp,case,relm,case_id,regnam,period,fig_path)

  del(data,dsub,nt,ny,nx)
  return mkclass

def KMeans_Clustering(da,season,n_cluster,mip,exp,relm,case,case_id,region,regnam,period,out_path,fig_path):
  '''
  da for one point in time (with lats x lons)
  '''
  l_global_norm = False 
  var = da.name
  vstr = da.name.upper()
  vunt = da.units.values
  print(var,vstr,vunt)

  if season == "DJF": 
    dsub = da.sel(time=da.time.dt.month.isin([1,2,12]), drop=True)
  elif season == "JJA":
    dsub = da.sel(time=da.time.dt.month.isin([6,7,8]), drop=True)
  elif season == "SON":
    dsub = da.sel(time=da.time.dt.month.isin([9,10,11]), drop=True)
  elif season == "MAM":
    dsub = da.sel(time=da.time.dt.month.isin([3,4,5]), drop=True)
  else: 
    #use all avaiable data   
    dsub = da 

  data = dsub.values
  nt,ny,nx = data.shape
  data = np.reshape(data, [nt, ny*nx], order='F')

  #construct mesh for data 
  xx = np.arange(nt)
  yy = np.arange(ny*nx)
  xx, yy = np.meshgrid(xx,yy)

  #scaling the data if needed 
  if l_global_norm: 
    x_norm = StandardScaler().fit_transform(data)
    #x_norm = normalize(data, norm="l2")
  else: 
    x_norm = data 
  #x_norm = norm_minmax(data)
  print(x_norm.max(),x_norm.min())

  for n in n_cluster[:]:
    #pca = PCA(n_components=n).fit(x_norm)
    #evar = pca.explained_variance_ratio_
    #clusterer = KMeans(init=pca.components_,n_clusters=n,n_init=1)
    #mkclass = clusterer.fit(x_norm)
    #print(82 * "_")
    #print("init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette")
    #bench_k_means(kmeans=clusterer, name="PCA-based", data=x_norm, labels=mkclass.labels_)
    #print(82 * "_")
    #reduced_data = PCA(n_components=n).fit_transform(data)
    #mkclass = clusterer.fit(reduced_data)
    #print(pca)
    #exit()

    clusterer = KMeans(init="k-means++",n_clusters=n,n_init=5,max_iter=100,random_state=0)
    mkclass = clusterer.fit(x_norm)

    #save data to nc file 
    #regimes = ['NAO$^-$', 'NAO$^+$', 'Blocking', 'Atlantic Ridge']
    #regimes = ['Regime 1','Regime 2','Regime 3','Regime 4','Regime 5','Regime 6']
    regimes = []
    for i in range(n):
      regimes.append('Regime {}'.format(i+1))
      onecen = mkclass.cluster_centers_[i,:].reshape(ny,nx, order='F')
      dtmp = xr.DataArray(
              data=np.array(onecen),
              name=regimes[i].replace(" ",""),
              dims=['lat','lon'],
              coords=dict(
                  lat=(["lat"],da.latitude.data),
                  lon=(["lon"],da.longitude.data),
                  ),
              attrs=dict(
                  description="KMean Clastering",
                  units=str(vunt),
                  ),
              )
      #merge data to single file
      if i == 0:
        dso = dtmp
      else:
        dso = xr.merge([dso,dtmp],compat="identical", join="inner")
      del(dtmp,onecen)
    
    if not os.path.exists(out_path):
      os.makedirs(out_path)
    fout_name = "{}_{}_{}_{}_{}_{}_{}clusters_{}_{}.nc".format(mip,exp,case,relm,case_id,vstr,n,regnam,period)
    dso.to_netcdf(os.path.join(out_path,fout_name)) #,encoding={'FillValue': None})

    #The silhouette_score gives the average value for all the samples.
    cluster_labels = clusterer.fit_predict(x_norm)
    silhouette_avg = silhouette_score(x_norm, cluster_labels)
    print("For n_clusters =", n,
          "The average silhouette_score is :",silhouette_avg,)
    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(x_norm, cluster_labels)
   
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)
    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(x_norm) + (n + 1) * 10])
    y_lower = 10
    for i in range(n):
      # Aggregate the silhouette scores for samples belonging to
      # cluster i, and sort them
      ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
      ith_cluster_silhouette_values.sort()
      size_cluster_i = ith_cluster_silhouette_values.shape[0]
      y_upper = y_lower + size_cluster_i
      color = cm.nipy_spectral(float(i) / n)
      ax1.fill_betweenx(
          np.arange(y_lower, y_upper),
          0,
          ith_cluster_silhouette_values,
          facecolor=color,
          edgecolor=color,
          alpha=0.7,
      )
      # Label the silhouette plots with their cluster numbers at the middle
      ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
      # Compute the new y_lower for next plot
      y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n)
    #ax2.scatter( xx, yy, c = x_norm[xx,yy], marker=".", s=30, lw=0, alpha=0.7, edgecolor="k")
    ax2.scatter(
        x_norm[:, 0], x_norm[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
    )

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="o",
        c="white",
        alpha=1,
        s=200,
        edgecolor="k",
    )

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

    ax2.set_title("The clustered data at points 1 & 2")
    ax2.set_xlabel("Feature space for the 1st point(space)")
    ax2.set_ylabel("Feature space for the 2nd point(space)")

    plt.suptitle(
        "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
        % n,
        fontsize=14,
        fontweight="bold",
    )

    if not os.path.exists(fig_path):
      os.makedirs(fig_path)
    fig_name = 'Silhouette_score_{}clusters_{}_{}_{}_{}_{}_{}_{}.pdf'.format(
                n,mip,exp,case,relm,case_id,regnam,period)
    plt.savefig(os.path.join(fig_path, fig_name),dpi=300)
    plt.close()

    #Visualize weather regimes
    print(mkclass)
    draw_weather_regime_map(dsub,mkclass,n,mip,exp,relm,case,case_id,region,regnam,period,fig_path)

  del(data,dsub,nt,ny,nx)
  exit()
  return mkclass

def draw_silhouette_analysis_similarity(ss,sim,mip,exp,case,relm,case_id,regnam,period,fig_dir):
  import seaborn as sns
  xn = normalize1(ss)
  x1 = xn[np.triu_indices(xn.shape[0],k=1)]
  
  color = {'md':   'y',
           'str':  'c',
           'ed':   'm',
           'ssim': 'k'}

  label = {'md':   'MD',
           'str':  'STR',
           'ed':   'ED', 
           'ssim': 'SSIM'}

  fig = plt.figure(figsize=(3,3))
  ax = plt.axes([.15,.15,.8,.8])
  
  sns.distplot(x1,hist=False,kde=True,ax=ax,norm_hist=False,
       bins=np.arange(0.,1.001,0.02), color = 'k',
       hist_kws={'edgecolor':'none', 'alpha':.2},
       kde_kws={'linewidth': 2, 'alpha':.8}, label=label[sim])

  sns.distplot(x1,hist=True,kde=True,ax=ax,norm_hist=False,
       bins=np.arange(0.,1.001,0.015), color=color[sim],
       hist_kws={'edgecolor':'none', 'alpha':.1},
       kde_kws={'linewidth': 0, 'alpha':.5}, label='')

  #plt.legend(loc=2, frameon=False)
  ax.set_ylabel('Probability distribution', fontsize=12)
  ax.set_xlabel('Normalized similarity', fontsize=12)
  
  ax.spines['right'].set_color('none')
  ax.spines['top'].set_color('none')
  ax.xaxis.set_ticks_position('bottom')
  ax.spines['bottom'].set_position(('axes', -0.0))
  ax.yaxis.set_ticks_position('left')
  ax.spines['left'].set_position(('axes', -0.0))
  ax.set_xlim([0, 1.01])
  ax.xaxis.grid(False)
  ax.yaxis.grid(False)
  ax.ticklabel_format(style='plain')
  #ax.ticklabel_format(useOffset=False)
  ax.ticklabel_format(axis='y', style='sci', scilimits=(-4,4))
  plt.xticks([0,.5,1])
  plt.yticks([],[])

  if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)
  fig_name = 'S-D_{}_{}_{}_{}_{}_{}_{}.pdf'.format(mip,exp,case,relm,case_id,regnam,period)
  plt.savefig(os.path.join(fig_dir, fig_name),dpi=300)
  plt.close()

  return 

def draw_silhouette_analysis_matrix(ss,sim,mip,exp,case,relm,case_id,regnam,period,fig_dir):
  import seaborn as sns
  xn = normalize1(ss)

  fig = plt.figure(figsize=[5,4])

  mask = np.zeros_like(xn)
  mask[np.triu_indices_from(mask)] = True

  ax = plt.axes([.1,.1, .8,.8])

  ax = sns.heatmap(pd.DataFrame(xn[:100,:100]),
                   #annot=True,
                   #mask = mask,
                   #cmap="YlGnBu",
                   vmin = 0.,
                   vmax=1.,
                   cbar_kws={'label': 'Normalized similarity', "shrink": .75},
                   #annot_kws={'size': 6},
                   #fmt=".1f", center=.1,
                   #cbar = cbar,
                   #linewidths=.25,
                   #linecolor='gray'
                   )

  ax.axhline(y=0, color='gray',linewidth=.25)
  plt.xticks( [],[])
  plt.yticks( [],[])
  
  if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)
  fig_name = 'S-matrix_{}_{}_{}_{}_{}_{}_{}.pdf'.format(mip,exp,case,relm,case_id,regnam,period)
  plt.savefig(os.path.join(fig_dir, fig_name),dpi=300)
  plt.close()
  
  return 

def draw_clustering_analysis(sim,mkclass,x,df,mip,exp,case,relm,case_id,regnam,period,fig_dir):
  import seaborn as sns
  
  if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

  colors = ['tomato', 'lightgreen', 'royalblue', 'y', 'c', 'm']
  clusters_all = mkclass['Clustering_all']
  for i, C in enumerate( mkclass['C_all'][:] ):
    fig = plt.figure(figsize=(4,4))
    ax = plt.axes([.15,.15,.8,.8])
    #plt.scatter(x[:,0],x[:,1],alpha=.6,color='r',s=15)
    print(i,C)
    exit()
    for kk in np.unique(clusters_all[i]):
        pp = np.array([x[j] for j in range(len(x)) if clusters_all[i][j] == kk])
        ax.scatter(pp[:, 0], pp[:, 1], s=10, c=colors[int(kk)], alpha = .8)
    #plt.scatter(x[:,0],x[:,1],alpha=.6,color='r',s=15)
    plt.scatter(C[:,0],C[:,1],marker='*',color='r',s=100,alpha=0.9)
    ax.set_ylim(-.1,1.1)
    ax.set_xlim(-.1,1.1)
    ax.set_xlabel('Dim-1', fontsize=14)
    ax.set_ylabel('Dim-2', fontsize=14)
    xlab = np.arange(0,1.1,.2)
    xlabs = ['%.1f' % a for a in xlab]
    plt.yticks(xlab,xlabs, fontsize = 10)
    plt.xticks(xlab,xlabs, fontsize = 10)
    fig_nam1 = 'Scatter_Clust{:02d}_{}_{}_{}_{}_{}_{}_{}.pdf'.format(
                              i+1,mip,exp,case,relm,case_id,regnam,period)
    fig.savefig(os.path.join(fig_dir,fig_nam1),dpi=300)

  fig = plt.figure(figsize=(3,3))
  ax = plt.axes([.15,.15,.8,.8])
  Plot_SS(df, colrs=colors[:4], ax = ax)
  fig_nam2 = 'Silhouete_{}_{}_{}_{}_{}_{}_{}.pdf'.format(mip,exp,case,relm,case_id,regnam,period)
  fig.savefig(os.path.join(fig_dir,fig_nam2),dpi=300)
  plt.close()

  #video = odir+'video.mp4'
  #try:
  #    os.system('rm '+video)
  #    os.system('ffmpeg -r 2 -f image2 -s 1920x1080 -i '+odir+'/%02d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p '+video)
  #except:
  #    print('fail to convert png to video')

  return

def draw_clustering_uncertainty(ss,mip,exp,case,relm,case_id,regnam,period,fig_dir):
  from sklearn import metrics as mtr
  import seaborn as sns

  ifiles = sorted(glob.glob(odir0+'/run_*/cluster.csv'))
  xx = pd.DataFrame({ int(f.split('run_')[-1].split('/')[0]) : 
                      pd.read_csv(f, index_col=0).values[:,0] 
                      for f in ifiles 
                      })
  xx['rnd'] = np.random.randint(k, size=len(xx) )
  columns = xx.columns
  do = pd.DataFrame(index=columns, columns = columns)

  for c1 in columns:
      for c2 in columns:
          print(c1, c2)
          d1, d2 = xx[c1].values, xx[c2].values
          do.loc[c1, c2] = 1 - mtr.adjusted_mutual_info_score(d1, d2)

  fout = 'CUD_{}_{}_{}_{}_{}_{}_{}.csv'.format(mip,exp,case,relm,case_id,regnam,period)
  do.to_csv(os.path.join(fig_dir,fout))

  do1 = do.astype(float).abs()
  mask = np.zeros_like(do1.values)
  mask[np.triu_indices_from(mask)] = True

  #figure 
  fig = plt.figure(figsize=[5,4])
  ax = plt.axes([.15,.15,.8,.8])
  ax = sns.heatmap(do1,
                   annot=True,
                   mask = mask,
                   vmin = 0.,
                   vmax=1.,
                   cbar_kws={'label': 'CUD', "shrink": .5},
                   annot_kws={'size': 6},
                   fmt=".2f", center=.1,
                   linewidths=.25,
                   linecolor='gray'
                   )

  ax.axhline(y=0, color='gray',linewidth=.25)
  ax.axhline(y=do1.shape[1], color='gray',linewidth= 1)
  ax.axvline(x=0, color='gray',linewidth=.25)
  ax.axvline(x=do1.shape[0], color='gray',linewidth=1)

  ax.figure.axes[-1].yaxis.label.set_size(13)
  plt.xticks( rotation=90, fontsize=9)
  plt.yticks( rotation=0, fontsize=9)

  if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)
  fig_name = 'CUD_{}_{}_{}_{}_{}_{}_{}.pdf'.format(mip,exp,case,relm,case_id,regnam,period)
  fig.savefig(os.path.join(fig_dir,fig_name),dpi=300)
  
  return 

def norm_minmax(x):
    return ( x - x.min() ) / ( x.max() - x.min())

def denorm_minmax(x,y):
    return y * ( x.max() - x.min()) + x.min() 

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
  out_path = os.path.join("/lcrc/group/e3sm/public_html/diagnostic_output/ac.szhang/polar_analysis/data","weather_regime","raw_index")
  fig_path = os.path.join("/lcrc/group/e3sm/public_html/diagnostic_output/ac.szhang/polar_analysis/figure", "weather_regimes","polar_region","figure")

  # region of interest (nino sector)
  regions   = {'socn':{'west':0., 'east':360., 'south':-90., 'north':-45.}}
  
  mip       = 'analysis'
  exp       = 'historical'
  relm      = 'en00'
  case_id   = 'weather_regime'
  tableId   = "Amon"
  products  = ["NOAA_20C","ERA5"]
  var       = "tas"
  varstr    = "TREFHT"
  period    = "195001-201412"
  run_path  = "/lcrc/group/e3sm/ac.szhang/acme_scratch/data/pcmdi/model/monthly"
  run_mask  = "/lcrc/group/e3sm/ac.szhang/acme_scratch/data/pcmdi/model/fixed/sftlf"
  season    = "Monthly"
  n_cluster = np.arange(2,21,2)
  use_sk_kmean_don = False 
  use_sk_kmean_don = True 

  for regnam in regions.keys(): 
    region = regions[regnam]  
    for i,product in enumerate(products):
      case = product
      ptmp = os.path.join(run_path,mip,case,tableId,var)
      ftmp = '{}.{}.{}.{}.*.{}.{}.nc'.format(mip,exp,product,relm,var,"*")
      filePath = sorted(glob.glob(os.path.join(ptmp,ftmp)))
      if len(filePath) > 0 and os.path.isfile(filePath[0]):
        data_fil = filePath[0]
        mask_fil = os.path.join(run_mask,tableId,mip+"."+exp+"."+case+".fx.sftlf.nc")
        case_dict = collections.OrderedDict()
        if case == "ERA5_HRES":
          case_dict[case] = Case(data_fil, var=var, color="blue", label=case)
        else:
          case_dict[case] = Case(data_fil, var=varstr, color="blue", label=case)
        case_dict['mask'] = Case(mask_fil, var="sftlf", color="blue", label=case)

        #call fuction to generate weather regime analysis 
        main(fig_path,out_path,mip,exp,relm,case_dict,case_id,region,regnam,period,season,use_sk_kmean_don,n_cluster)
