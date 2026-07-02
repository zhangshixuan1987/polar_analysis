import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from polar_utils.common import (
    Case,
    low_pass,
    detrend_dim,
    slice_region,
    pearson_r_p_value,
    draw_regression_map,
    draw_regional_box,
    open_dataset
)

# ============================================================
# Core K-Means Weather Regimes Analysis
# ============================================================
def run_weather_regime_analysis(fig_path, out_path, mip, exp, relm, case_id, period, case_dict, region, regnam, season, n_clusters=[4]):
    for key in case_dict:
        case = key
        var = case_dict[key].var
        data = case_dict[key].path
        
    print("working on Weather Regime analysis for:", case, var)
    ds = open_dataset(data)
    ymds = '{}-{}-01'.format(period.split("-")[0][0:4], period.split("-")[0][4:6])
    ymde = '{}-{}-31'.format(period.split("-")[1][0:4], period.split("-")[1][4:6])
    ds = ds.sel(time=slice(ymds, ymde))
    
    lat_name = 'latitude' if 'latitude' in ds.dims else ('lat' if 'lat' in ds.dims else None)
    lon_name = 'longitude' if 'longitude' in ds.dims else ('lon' if 'lon' in ds.dims else None)
    if lat_name and lon_name and (lat_name == 'lat' or lon_name == 'lon'):
        ds = ds.rename({lat_name: 'latitude', lon_name: 'longitude'})
        
    da = ds[var]
    if da.units == "Pa":
        da = da / 100.
        da = da.assign_attrs(units='hPa')
        
    # Season slice
    if season == "DJF":
        da_season = da.sel(time=da.time.dt.month.isin([1, 2, 12]))
    elif season == "JJA":
        da_season = da.sel(time=da.time.dt.month.isin([6, 7, 8]))
    elif season == "SON":
        da_season = da.sel(time=da.time.dt.month.isin([9, 10, 11]))
    elif season == "MAM":
        da_season = da.sel(time=da.time.dt.month.isin([3, 4, 5]))
    else:
        da_season = da
        
    # Region slice
    lat_slice = slice(region['south'], region['north'])
    lon_slice = slice(region['west'], region['east'])
    da_reg = da_season.sel(latitude=lat_slice, longitude=lon_slice)
    
    # Pre-process: Weight anomalies by sqrt(cos(lat))
    clat = da_reg['latitude'].astype(np.float64)
    clat = np.sqrt(np.cos(np.deg2rad(clat)))
    da_wgt = da_reg * clat
    
    nt, ny, nx = da_wgt.shape
    X_flat = da_wgt.values.reshape(nt, -1, order='F')
    
    # Drop NaN features
    valid_mask = ~np.isnan(X_flat).any(axis=0)
    X_clean = X_flat[:, valid_mask]
    
    # Normalize
    X_scaled = StandardScaler().fit_transform(X_clean)
    
    for n in n_clusters:
        clusterer = KMeans(init="k-means++", n_clusters=n, n_init=5, max_iter=100, random_state=0)
        labels = clusterer.fit_predict(X_scaled)
        centers = clusterer.cluster_centers_
        
        # Reconstruct centers
        centers_full = np.full((n, X_flat.shape[1]), np.nan)
        centers_full[:, valid_mask] = centers
        centers_3d = centers_full.reshape(n, ny, nx, order='F')
        
        # Plot regime maps
        fig = plt.figure(figsize=(12, 10))
        for i in range(n):
            ax = fig.add_subplot(2, 2, i + 1)
            im = ax.pcolormesh(da_reg.longitude, da_reg.latitude, centers_3d[i, :, :], cmap='RdYlBu_r', shading='auto')
            plt.colorbar(im, ax=ax)
            ax.set_title('Regime {}'.format(i + 1))
            
        plt.suptitle('Weather Regimes for {} ({}, {})'.format(var, regnam, season))
        plt.tight_layout()
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
        fig_name = "fig_regime_{}_{}_{}_{}_{}_{}_{}.pdf".format(regnam, season, mip, exp, case, case_id, period)
        plt.savefig(os.path.join(fig_path, fig_name))
        plt.close()
        
        # Save output NetCDF
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        out_file = 'Regime_{}_{}_{}_{}_{}_{}_{}.nc'.format(regnam, season, mip, exp, case, case_id, period)
        ds_out = xr.Dataset(
            {
                "regimes": xr.DataArray(centers_3d, dims=['regime', 'latitude', 'longitude'], coords={'regime': np.arange(n), 'latitude': da_reg.latitude, 'longitude': da_reg.longitude}),
                "labels": xr.DataArray(labels, dims=['time'], coords={'time': da_reg.time})
            }
        )
        ds_out.to_netcdf(os.path.join(out_path, out_file))
