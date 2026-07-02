import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import linregress

from polar_utils.common import (
    Case,
    low_pass,
    detrend_dim,
    slice_region,
    pearson_r_p_value,
    draw_regression_map,
    draw_regional_box
)
from polar_utils.eof import eofunc_eofs_svd, eofunc_pcs_svd

# ============================================================
# Core Trend Analysis
# ============================================================
def calculate_grid_trend(da):
    # da has shape (time, lat, lon)
    ntime, nlat, nlon = da.shape
    x = np.arange(ntime)
    
    flat_da = da.values.reshape(ntime, -1)
    slopes = np.full(flat_da.shape[1], np.nan)
    p_values = np.full(flat_da.shape[1], np.nan)
    
    valid_mask = ~np.isnan(flat_da).any(axis=0)
    for idx in np.where(valid_mask)[0]:
        slope, intercept, r_value, p_value, std_err = linregress(x, flat_da[:, idx])
        slopes[idx] = slope * 12.0 * 10.0 # convert to trend per decade
        p_values[idx] = p_value
        
    trend_map = slopes.reshape(nlat, nlon)
    pval_map = p_values.reshape(nlat, nlon)
    
    return trend_map, pval_map

def run_trend_analysis(fig_path, out_path, mip, exp, relm, case_id, period, pclimo, case_dict, region, regnam, season):
    for key in case_dict:
        case = key
        var = case_dict[key].var
        data = case_dict[key].path
        
    print("working on Trend analysis for:", case, var)
    ds = xr.open_dataset(data)
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
        
    # Region Slicing
    lat_slice = slice(region['south'], region['north'])
    lon_slice = slice(region['west'], region['east'])
    da_reg = da.sel(latitude=lat_slice, longitude=lon_slice)
    
    # Calculate grid-wise trend
    trend_map, pval_map = calculate_grid_trend(da_reg)
    
    # Run EOF of detrended anomalies (Trend-Uncorrelated EOFs)
    clat = da_reg['latitude'].astype(np.float64)
    clat = np.sqrt(np.cos(np.deg2rad(clat)))
    
    da_detrend = detrend_dim(da_reg, 'time', 1)
    da_wgt = da_detrend * clat
    
    neofs = 4
    eofs = eofunc_eofs_svd(da_wgt.values, neofs=neofs)
    pcs = eofunc_pcs_svd(da_wgt.values, npcs=neofs)
    pcs_std = pcs / pcs.std(dim='time')
    
    # Plot Trend Map
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.pcolormesh(da_reg.longitude, da_reg.latitude, trend_map, cmap='RdYlBu_r', shading='auto')
    plt.colorbar(im, ax=ax, label='Trend (per decade)')
    ax.set_title('Linear Trend of {} ({})'.format(var, regnam))
    
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    fig_name = "fig_trend_map_{}_{}_{}_{}_{}_{}_{}.pdf".format(regnam, season, mip, exp, case, case_id, period)
    plt.savefig(os.path.join(fig_path, fig_name))
    plt.close()
    
    # Save outputs
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    out_file = 'Trend_{}_{}_{}_{}_{}_{}_{}.nc'.format(regnam, season, mip, exp, case, case_id, period)
    ds_out = xr.Dataset(
        {
            "trend": xr.DataArray(trend_map, dims=['latitude', 'longitude'], coords={'latitude': da_reg.latitude, 'longitude': da_reg.longitude}),
            "p_value": xr.DataArray(pval_map, dims=['latitude', 'longitude'], coords={'latitude': da_reg.latitude, 'longitude': da_reg.longitude}),
            "eofs": eofs,
            "pcs": pcs_std
        }
    )
    ds_out.to_netcdf(os.path.join(out_path, out_file))
