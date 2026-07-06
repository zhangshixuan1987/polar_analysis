import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.linalg import svd

from .common import (
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
# Custom SVD-based EOF implementation (geocat fallback)
# ============================================================
def eofunc_eofs_svd(data, neofs=15):
    # data has shape (time, lat, lon)
    ntime, nlat, nlon = data.shape
    flat_data = data.reshape(ntime, -1)
    
    # Identify non-NaN spatial points
    valid_mask = ~np.isnan(flat_data).any(axis=0)
    X = flat_data[:, valid_mask]
    
    # Center the data (anomalies)
    X_mean = X.mean(axis=0)
    X_anom = X - X_mean
    
    # Run SVD
    U, s, Vt = svd(X_anom, full_matrices=False)
    
    # Calculate variance fraction
    var_frac = (s**2) / np.sum(s**2)
    
    # Reconstruct spatial EOFs
    eofs_flat = np.full((neofs, flat_data.shape[1]), np.nan)
    eofs_flat[:, valid_mask] = Vt[:neofs, :]
    eofs = eofs_flat.reshape(neofs, nlat, nlon)
    
    # Wrap in xarray or custom object with attributes
    eofs_da = xr.DataArray(eofs, dims=['eof', 'latitude', 'longitude'])
    eofs_da.attrs['varianceFraction'] = var_frac[:neofs]
    return eofs_da

def eofunc_pcs_svd(data, npcs=15):
    ntime, nlat, nlon = data.shape
    flat_data = data.reshape(ntime, -1)
    
    valid_mask = ~np.isnan(flat_data).any(axis=0)
    X = flat_data[:, valid_mask]
    X_mean = X.mean(axis=0)
    X_anom = X - X_mean
    
    U, s, Vt = svd(X_anom, full_matrices=False)
    
    pcs = U[:, :npcs] * s[:npcs]
    pcs_da = xr.DataArray(pcs, dims=['time', 'pc'])
    
    var_frac = (s**2) / np.sum(s**2)
    pcs_da.attrs['varianceFraction'] = var_frac[:npcs]
    return pcs_da

# ============================================================
# Seasonal/Annual Mean calculation helpers
# ============================================================
def season_mean(ds, calendar="standard"):
    return ds.groupby('time.season').mean(dim='time')

def season_mean2(ds, season, calendar="standard"):
    return ds.sel(time=ds.time.dt.season == season).groupby('time.year').mean(dim='time')

def annual_mean(ds, calendar="standard"):
    return ds.groupby('time.year').mean(dim='time')

# ============================================================
# Main EOF Analysis routines
# ============================================================
def run_eof_analysis(fig_path, out_path, mip, exp, relm, case_id, period, pclimo, case_dict, region, regnam, season):
    for key in case_dict:
        case = key
        var = case_dict[key].var
        data = case_dict[key].path
        
    print("working on EOF analysis for:", case, var)
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
        
    # Standard Lat/Lon Slicing
    lat_slice = slice(region['south'], region['north'])
    lon_slice = slice(region['west'], region['east'])
    da_reg = da.sel(latitude=lat_slice, longitude=lon_slice)
    
    # Weight anomalies by sqrt(cos(lat))
    clat = da_reg['latitude'].astype(np.float64)
    clat = np.sqrt(np.cos(np.deg2rad(clat)))
    da_wgt = da_reg * clat
    da_wgt.attrs = da_reg.attrs
    da_wgt.attrs['name'] = var
    
    # Run EOF & PCs via SVD
    neofs = 4
    eofs = eofunc_eofs_svd(da_wgt.values, neofs=neofs)
    pcs = eofunc_pcs_svd(da_wgt.values, npcs=neofs)
    
    # Standardize PCs
    pcs_std = pcs / pcs.std(dim='time')
    
    # Draw results
    fig = plt.figure(figsize=(12, 10))
    for i in range(neofs):
        ax = fig.add_subplot(2, 2, i + 1)
        im = ax.pcolormesh(da_reg.longitude, da_reg.latitude, eofs[i, :, :], cmap='RdYlBu_r', shading='auto')
        plt.colorbar(im, ax=ax)
        ax.set_title('EOF {} ({:0.1f}%)'.format(i + 1, eofs.attrs['varianceFraction'][i] * 100))
        
    plt.suptitle('EOF Analysis for {} ({}, {})'.format(var, regnam, season))
    plt.tight_layout()
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    fig_name = "fig_eof_{}_{}_{}_{}_{}_{}_{}.pdf".format(regnam, season, mip, exp, case, case_id, period)
    plt.savefig(os.path.join(fig_path, fig_name))
    plt.close()
    
    # Save outputs as netcdf
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    out_file = 'EOF_{}_{}_{}_{}_{}_{}_{}.nc'.format(regnam, season, mip, exp, case, case_id, period)
    ds_out = xr.Dataset(
        {
            "eofs": eofs,
            "pcs": pcs_std
        }
    )
    ds_out.to_netcdf(os.path.join(out_path, out_file))
