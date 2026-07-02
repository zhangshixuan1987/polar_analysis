import os
import glob
import collections
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from polar_utils.common import (
    Case,
    low_pass,
    detrend_dim,
    slice_region,
    pearson_r_p_value,
    draw_regression_map,
    draw_regional_box
)

# ============================================================
# Weighted area average
# ============================================================
def wgt_areaave(da, region):
    lat_name = 'latitude' if 'latitude' in da.dims else ('lat' if 'lat' in da.dims else None)
    lon_name = 'longitude' if 'longitude' in da.dims else ('lon' if 'lon' in da.dims else None)
    lons = da[lon_name]
    lats = da[lat_name]
    
    latN = region['north']
    latS = region['south']
    lonE = region['east']
    lonW = region['west']
    
    if (((lonW < 0) or (lonE < 0)) and (lons.values.min() > -1)):
        da = da.assign_coords(**{lon_name: ((lons + 180) % 360 - 180)})
        lons = ((lons + 180) % 360 - 180)
        
    iplat = lats.where((lats >= latS) & (lats <= latN), drop=True)
    iplon = lons.where((lons >= lonW) & (lons <= lonE), drop=True)
    
    wgt = np.cos(np.deg2rad(lats))
    odat = da.sel(**{lat_name: iplat, lon_name: iplon}).weighted(wgt).mean((lon_name, lat_name), skipna=True)
    return odat

# ============================================================
# Define Nino index
# ============================================================
def define_nino(mip, exp, relm, case, case_id, period, vstr, vunt, da, region, out_path):
    times = da.time.dt.strftime("%Y-%m-%d")
    time_str = da.time.dt.strftime("%Y-%m-%d")
    ntime = len(times)
    print("number of total months in Nino data: ", ntime)

    clm = da.groupby('time.month').mean(dim='time')
    anm = (da.groupby('time.month') - clm)
    dclm = anm.copy()
    for i in range(len(clm)):
        dclm[i::12, :, :] = clm[i, :, :].copy()

    nino_sst = wgt_areaave(da, region)
    nino_clm = wgt_areaave(dclm, region)
    nino_ind = wgt_areaave(anm, region)
    ninoSD_ind = nino_ind / nino_ind.std(dim='time')
    
    rnino_ind = low_pass(1.0 / 5.0, nino_ind, axis=0)
    onino_ind = low_pass(1.0 / 3.0, nino_ind, axis=0)
    rninoSD_ind = low_pass(1.0 / 5.0, ninoSD_ind, axis=0)
    oninoSD_ind = low_pass(1.0 / 3.0, ninoSD_ind, axis=0)

    colnams = ['time', 'ninosst_val(degC)', 'ninosst_clm(degC)', 'nino_idx(degC)', 'rnino_idx(degC)',
               'onino_idx(degC)', 'ninoSD_idx(1)', 'rninoSD_idx(1)', 'oninoSD_idx(1)']
    df = pd.DataFrame([], columns=colnams)
    df['time'] = time_str
    df['ninosst_val(degC)'] = nino_sst
    df['ninosst_clm(degC)'] = nino_clm
    df['nino_idx(degC)'] = nino_ind
    df['ninoSD_idx(1)'] = ninoSD_ind
    df['rnino_idx(degC)'] = rnino_ind
    df['rninoSD_idx(1)'] = rninoSD_ind
    df['onino_idx(degC)'] = onino_ind
    df['oninoSD_idx(1)'] = oninoSD_ind

    df = df.reset_index(drop=True)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    out_file = '{}.{}.{}.{}.{}.{}.{}.csv'.format(mip, exp, case, relm, case_id, vstr, period)
    df.to_csv(os.path.join(out_path, out_file), index=False)
    return df, colnams

# ============================================================
# Plot Nino index Time Series
# ============================================================
def draw_nino_ts(fig_path, nino_df, colnams, nino_region, mip, exp, relm, case, case_id, period, var, vunt):
    time = nino_df['time']
    xtime = np.linspace(1, len(time), len(time))
    years = int(time[0].split("-")[0])

    xtick = np.arange(0, len(time))
    xlabs = np.arange(0, len(time)) / 12.0 + years
    fontsize = 16
    fig = plt.figure(figsize=(8, 11))
    
    plot_cols = [c for c in colnams if c != 'time']
    for i, col in enumerate(plot_cols[:6]):
        var0 = np.array(nino_df[col])
        var1 = low_pass(1.0 / 11.0, var0, axis=0)
        ax = fig.add_subplot(min(len(plot_cols), 6), 1, i + 1)
        ax.plot(xtime, var0, color='grey', alpha=1.0, linewidth=0.8, label='monthly')
        ax.plot(xtime, var1, color='black', alpha=1.0, linewidth=1.4, label='11-point Hamming')
        ax.set_xlabel('Time (years)')
        ax.set_ylabel(col)
        ax.set_xlim(0, len(time))
        ax.set_xticks(xtick[::120])
        ax.set_xticklabels(xlabs[::120].astype(int))
        ax.grid(True)
        if i + 1 == min(len(plot_cols), 6):
            ax.legend(loc='lower right', prop={'size': 8})
            
    plt.suptitle('Nino indices ({},{})'.format(var, vunt), fontsize=fontsize * 1.1)
    plt.tight_layout()
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    fig_name = "fig_ts_{}_{}_{}_{}_{}_{}_{}.pdf".format(mip, exp, case, relm, case_id, var, period)
    plt.savefig(os.path.join(fig_path, fig_name))
    plt.close()

# ============================================================
# Helper: Save regression data to NetCDF for Nino
# ============================================================
def save_nino_regression_data(vcor, vreg, pval, vsig, mip, exp, relm, case, case_id, period, var, vunt, out_path):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    out_file = "{}.{}.{}.{}.{}.{}.{}.nc".format(mip, exp, case, relm, case_id, var, period)
    ds = xr.Dataset(
        {
            "correlation": vcor,
            "regression": vreg,
            "p_value": pval,
            "sig_regression": vsig,
        }
    )
    ds.to_netcdf(os.path.join(out_path, out_file))

# ============================================================
# Plot Nino regression map
# ============================================================
def draw_nino_map(out_path, fig_path, da, nino_df, colnams, nino_region, mip, exp, relm, case, case_id, period, var, vunt):
    lat_name = 'latitude' if 'latitude' in da.dims else 'lat'
    lon_name = 'longitude' if 'longitude' in da.dims else 'lon'
    lons = da[lon_name][:]
    lats = da[lat_name][:]
    clm = da.groupby('time.month').mean(dim='time')
    anm = (da.groupby('time.month') - clm)
    vstr = da.name
    vunt = da.units
    
    time = nino_df['time']
    nino_df = nino_df.set_index('time')
    nino_exist = False
    for col in colnams:
        if 'ninoSD_idx' in col:
            nino = nino_df[col].to_xarray()
            nino_exist = True
    if not nino_exist:
        raise ValueError('ninoSD_idx not exist, please check....')
        
    nino = nino.assign_coords({"time": da.time})
    ninoSD = nino / nino.std(dim='time')
    
    rninoSD = ninoSD.copy()
    ranm = anm.copy()
    rninoSD[:] = low_pass(1.0 / 5.0, ninoSD, axis=0)
    if np.isnan(anm).any():
        tmp1 = ranm.fillna(-99999)
        tmp1 = low_pass(1.0 / 5.0, tmp1, axis=0)
        ranm = anm.copy()
        ranm[:, :, :] = tmp1[:, :, :]
        ranm = ranm.where(ranm > -10000)
    else:
        ranm[:, :, :] = low_pass(1.0 / 5.0, anm[:, :, :], axis=0)
    rdanm = detrend_dim(ranm, 'time', 1)
    
    vcor = xr.corr(rninoSD, rdanm, dim="time")
    vreg = xr.cov(rninoSD, rdanm, dim="time") / rninoSD.var(dim='time', skipna=True).values
    pval = pearson_r_p_value(rninoSD, rdanm, dim="time")
    
    vsig = vreg.copy()
    vsig = vsig.where(pval <= 0.05)
    
    save_nino_regression_data(vcor, vreg, pval, vsig, mip, exp, relm, case, case_id, period, var, vunt, out_path)
    
    vmin = -1.0
    vmax = 1.0
    nlev = 21
    fontsize = 16
    bartitle = 'Regressed {}({})'.format(vstr, vunt)
    fig = plt.figure(figsize=(10, 12))
    grid = fig.add_gridspec(ncols=1, nrows=1)
    ax1, fill1 = draw_regression_map(vstr, vunt, lons, lats, vcor, vreg, pval, fig, nino_region,
                                     fontsize, 'Nino3.4-SST Pattern', grid[0, 0], vmin, vmax, nlev)
    cb = fig.colorbar(fill1, ax=[ax1], drawedges=True, orientation='horizontal',
                      shrink=0.95, aspect=40, pad=0.05, extendfrac='auto', extendrect=True)
    ticks = np.linspace(vmin, vmax, nlev)
    labels = []
    for i, tick in enumerate(ticks):
        if i % 2 == 0:
            labels.append('{:0.1f}'.format(tick))
        else:
            labels.append('')
    cb.set_ticks(ticks=ticks, labels=labels, fontsize=fontsize * 0.9)
    cb.set_label(label=bartitle, fontsize=fontsize * 0.95)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams.update({'font.size': fontsize})
    plt.draw()
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    fig_name = "fig_2d_map_{}_{}_{}_{}_{}_{}_{}.pdf".format(mip, exp, case, relm, case_id, var, period)
    plt.savefig(os.path.join(fig_path, fig_name))
    plt.close()

# ============================================================
# Main Nino Index Driver
# ============================================================
def run_nino_index_generation(fig_path, out_path, mip, exp, relm, case_id, period, case_dict, nino_region, l_check_nino_region):
    for key in case_dict:
        case = key
        var = case_dict[key].var
        data = case_dict[key].path
        
    print("working on Nino index generation for:", case, var)
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
        
    nino_df, colnams = define_nino(mip, exp, relm, case, case_id, period, "SST", "degC", da, nino_region, out_path)
    draw_nino_ts(fig_path, nino_df, colnams, nino_region, mip, exp, relm, case, case_id, period, "SST", "degC")
    draw_nino_map(out_path, fig_path, da, nino_df, colnams, nino_region, mip, exp, relm, case, case_id, period, "SST", "degC")
    return nino_df

# ============================================================
# Helper: Save lead-lag NetCDF for Nino
# ============================================================
def save_nino_leadlag_nc(corll, regll, pvlll, leadlag, mip, exp, relm, case, case_id, period, var, vunt, reg_idx, out_path):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    out_file = "{}_leadlag_{}.{}.{}.{}.{}.{}.{}.nc".format(reg_idx, mip, exp, case, relm, case_id, var, period)
    ds = xr.Dataset(
        {
            "correlation": xr.concat(corll, dim="lag"),
            "regression": xr.concat(regll, dim="lag"),
            "p_value": xr.concat(pvlll, dim="lag"),
        },
        coords={"lag": leadlag}
    )
    ds.to_netcdf(os.path.join(out_path, out_file))

# ============================================================
# Main Nino Lead-Lag Driver
# ============================================================
def nino_lead_lag_anl(mip, exp, relm, case, case_id, period, out_path, fig_path, da, nino, reg_idx, time_vals, var):
    lons = da['longitude'][:]
    lats = da['latitude'][:]
    clm = da.groupby('time.month').mean(dim='time')
    anm = (da.groupby('time.month') - clm)
    vstr = da.name
    vunt = da.units
    ninoSD = nino / nino.std(dim='time')
    
    rninoSD = ninoSD.copy()
    ranm = anm.copy()
    rninoSD[:] = low_pass(1.0 / 5.0, ninoSD, axis=0)
    if np.isnan(anm).any():
        tmp1 = ranm.fillna(-99999)
        tmp1 = low_pass(1.0 / 5.0, tmp1, axis=0)
        ranm = anm.copy()
        ranm[:, :, :] = tmp1[:, :, :]
        ranm = ranm.where(ranm > -10000)
    else:
        ranm[:, :, :] = low_pass(1.0 / 5.0, anm[:, :, :], axis=0)
    rdanm = detrend_dim(ranm, 'time', 1)
    
    leadlag = np.arange(-12, 18, 3)
    leadlag = leadlag[::-1]
    corll = []
    regll = []
    pvlll = []
    for tllg in leadlag:
        cor = xr.corr(rninoSD, rdanm.shift(time=tllg), dim="time")
        reg = xr.cov(rninoSD, rdanm.shift(time=tllg), dim="time") / rninoSD.var(dim='time', skipna=True).values
        pvl = pearson_r_p_value(rninoSD, rdanm.shift(time=tllg), dim="time")
        corll.append(cor)
        regll.append(reg)
        pvlll.append(pvl)
        
    save_nino_leadlag_nc(corll, regll, pvlll, leadlag, mip, exp, relm, case, case_id, period, var, vunt, reg_idx, out_path)
    
    vmax = 1.0
    vmin = -1.0
    nlev = 21
    ncol = 2
    nrow = int(len(leadlag) / ncol)
    fontsize = 16
    bartitle = 'Regressed {}({})'.format(vstr, vunt)
    fig = plt.figure(figsize=(16, 30))
    grid = fig.add_gridspec(ncols=ncol, nrows=nrow)
    axs = []
    fills = []
    
    nino_region = {'west': -180., 'east': 180., 'south': -90., 'north': -20.}
    for j in range(ncol):
        for i in range(nrow):
            k = i + j * nrow
            tllg = leadlag[k]
            if tllg > 0.0:
                tag = '{}-month {}'.format(abs(tllg), 'lag')
            elif tllg < 0.0:
                tag = '{}-month {}'.format(abs(tllg), 'lead')
            else:
                tag = 'Simultaneous'
            ax, fill = draw_regression_map(vstr, vunt, lons, lats, corll[k], regll[k], pvlll[k], fig, nino_region,
                                           fontsize, tag, grid[i, j], vmin, vmax, nlev)
            axs.append(ax)
            fills.append(fill)
            
    cb = fig.colorbar(fills[len(leadlag) - 1], ax=axs, drawedges=True, orientation='horizontal',
                      shrink=0.95, aspect=40, pad=0.05, extendfrac='auto', extendrect=True)
    ticks = np.linspace(vmin, vmax, nlev)
    labels = []
    for i, tick in enumerate(ticks):
        if i % 2 == 0:
            labels.append('{:0.1f}'.format(tick))
        else:
            labels.append('')
    cb.set_ticks(ticks=ticks, labels=labels, fontsize=fontsize * 0.9)
    cb.set_label(label=bartitle, fontsize=fontsize * 0.95)
    
    fig.suptitle('Corr. & Regress with Nino{}: {}({})'.format(reg_idx, vstr, vunt), fontsize=fontsize, y=0.9)
    plt.rcParams["font.family"] = "sans-serif"
    plt.draw()
    
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    fig_name = "fig_{}_leadlag_map_{}_{}_{}_{}_{}_{}_{}.pdf".format(reg_idx, mip, exp, case, relm, case_id, var, period)
    plt.savefig(os.path.join(fig_path, fig_name))
    plt.close()

def run_nino_leadlag_analysis(fig_path, out_path, mip, exp, relm, case_id, period, case_dict, nino_region, reg_idx, l_check_nino_region):
    for key in case_dict:
        case = key
        var = case_dict[key].var
        data = case_dict[key].path
        
    print("working on Nino lead-lag for:", case, var)
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
        
    nino_index_file = os.path.join(out_path.replace("lead_lag", "raw_index"), 
                                   "{}.{}.{}.{}.{}.{}.{}.csv".format(mip, exp, case, relm, case_id, "SST", period))
    nino_df = pd.read_csv(nino_index_file)
    nino_exist = False
    for col in nino_df.columns:
        if reg_idx in col:
            nino = nino_df[col].to_xarray()
            nino_exist = True
    if not nino_exist:
        raise ValueError('Index {} not exist, please check....'.format(reg_idx))
        
    nino = nino.assign_coords({"time": da.time})
    nino_lead_lag_anl(mip, exp, relm, case, case_id, period, out_path, fig_path, da, nino, reg_idx, da.time, var)
