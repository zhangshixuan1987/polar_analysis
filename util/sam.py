import os
import glob
import collections
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

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
# Helper: Find nearest coordinate index
# ============================================================
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

# ============================================================
# Define EOF-based SAM index
# ============================================================
def define_sam_eof(mip, exp, relm, case, case_id, period, vstr, vunt, da, out_path):
    times = da.time.dt.strftime("%Y-%m-%d")
    time_str = da.time.dt.strftime("%Y-%m-%d")
    ntime = len(times)
    print("number of total months in EOF SAM data: ", ntime)

    sam_evar    = da['pc'].values.copy()
    sam_evar[:] = da['frac'].values * 100.0
    sam_ind     = da['pc'].values.copy()
    samSD_ind   = sam_ind / sam_ind.std()
    
    rsam_ind    = low_pass(1.0 / 5.0, sam_ind, axis=0)
    osam_ind    = low_pass(1.0 / 3.0, sam_ind, axis=0)
    rsamSD_ind  = low_pass(1.0 / 5.0, samSD_ind, axis=0)
    osamSD_ind  = low_pass(1.0 / 3.0, samSD_ind, axis=0)

    colnams = ['time', 'sam_evar(%)', 'sam_idx(1)', 'rsam_idx(1)', 'osam_idx(1)', 'samSD_idx(1)', 'rsamSD_idx(1)', 'osamSD_idx(1)']
    df = pd.DataFrame([], columns=colnams)
    df['time'] = time_str
    df['sam_evar(%)'] = sam_evar
    df['sam_idx(1)'] = sam_ind
    df['samSD_idx(1)'] = samSD_ind
    df['rsam_idx(1)'] = rsam_ind
    df['rsamSD_idx(1)'] = rsamSD_ind
    df['osam_idx(1)'] = osam_ind
    df['osamSD_idx(1)'] = osamSD_ind

    df = df.reset_index(drop=True)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    out_file = '{}.{}.{}.{}.{}.{}.{}.csv'.format(mip, exp, case, relm, case_id, vstr, period)
    df.to_csv(os.path.join(out_path, out_file), index=False)
    return df, colnams

# ============================================================
# Define PSL-based SAM index (Gong & Wang 1999 definition)
# ============================================================
def define_sam_psl(mip, exp, relm, case, case_id, period, vstr, vunt, da, out_path):
    times = da.time.dt.strftime("%Y-%m-%d")
    time_str = da.time.dt.strftime("%Y-%m-%d")
    ntime = len(times)
    print("number of total months in PSL SAM data: ", ntime)

    lat_name = 'latitude' if 'latitude' in da.dims else 'lat'
    lon_name = 'longitude' if 'longitude' in da.dims else 'lon'
    lons = da[lon_name]
    lats = da[lat_name]
    
    north_lat, ilatn = find_nearest(lats.values, -40.0)
    south_lat, ilats = find_nearest(lats.values, -65.0)
    
    iplat = lats.where((lats == north_lat) | (lats == south_lat), drop=True)
    dsub = da.sel(**{lat_name: iplat}).mean(lon_name, skipna=True)

    clm = dsub.groupby('time.month').mean(dim='time')
    std = dsub.groupby('time.month').std(dim='time')
    anm = (dsub.groupby('time.month') - clm)
    norm = (dsub.groupby('time.month') - clm) / std
    
    dclm = anm.copy()
    for i in range(len(clm)):
        dclm[i::12, :] = clm[i, :].copy()
        
    sam_psl65S = dsub.sel(**{lat_name: south_lat}).values
    sam_psl40S = dsub.sel(**{lat_name: north_lat}).values
    sam_clm65S = dclm.sel(**{lat_name: south_lat}).values
    sam_clm40S = dclm.sel(**{lat_name: north_lat}).values
    sam_dpsl = anm.sel(**{lat_name: north_lat}).values - anm.sel(**{lat_name: south_lat}).values
    sam_ind = norm.sel(**{lat_name: north_lat}).values - norm.sel(**{lat_name: south_lat}).values
    samSD_ind = sam_ind / sam_ind.std()
    
    rsam_ind = low_pass(1.0 / 5.0, sam_ind, axis=0)
    osam_ind = low_pass(1.0 / 3.0, sam_ind, axis=0)
    rsamSD_ind = low_pass(1.0 / 5.0, samSD_ind, axis=0)
    osamSD_ind = low_pass(1.0 / 3.0, samSD_ind, axis=0)

    colnams = ['time', 'sam_psl65S(hPa)', 'sam_psl40S(hPa)', 'sam_clm65S(hPa)', 'sam_clm40S(hPa)', 'sam_dpsl(hPa)',
               'sam_idx(1)', 'rsam_idx(1)', 'osam_idx(1)', 'samSD_idx(1)', 'rsamSD_idx(1)', 'osamSD_idx(1)']
    df = pd.DataFrame([], columns=colnams)
    df['time'] = time_str
    df['sam_psl65S(hPa)'] = sam_psl65S
    df['sam_psl40S(hPa)'] = sam_psl40S
    df['sam_clm65S(hPa)'] = sam_clm65S
    df['sam_clm40S(hPa)'] = sam_clm40S
    df['sam_dpsl(hPa)'] = sam_dpsl
    df['sam_idx(1)'] = sam_ind
    df['rsam_idx(1)'] = rsam_ind
    df['osam_idx(1)'] = osam_ind
    df['samSD_idx(1)'] = samSD_ind
    df['rsamSD_idx(1)'] = rsamSD_ind
    df['osamSD_idx(1)'] = osamSD_ind

    df = df.reset_index(drop=True)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    out_file = '{}.{}.{}.{}.{}.{}.{}.csv'.format(mip, exp, case, relm, case_id, vstr, period)
    df.to_csv(os.path.join(out_path, out_file), index=False)
    return df, colnams

# ============================================================
# Draw SAM Index Time Series
# ============================================================
def draw_sam_ts(fig_path, sam_df, colnams, sam_region, mip, exp, relm, case, case_id, period, var, vunt):
    time = sam_df['time']
    xtime = np.linspace(1, len(time), len(time))
    years = int(time[0].split("-")[0])

    xtick = np.arange(0, len(time))
    xlabs = np.arange(0, len(time)) / 12.0 + years
    fontsize = 16
    fig = plt.figure(figsize=(8, 11))
    
    # We display up to 6 key variables in subplots
    plot_cols = [c for c in colnams if c != 'time']
    for i, col in enumerate(plot_cols[:6]):
        var0 = np.array(sam_df[col])
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
            
    plt.suptitle('SAM indices ({},{})'.format(var, vunt), fontsize=fontsize * 1.1)
    plt.tight_layout()
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    fig_name = "fig_ts_{}_{}_{}_{}_{}_{}_{}.pdf".format(mip, exp, case, relm, case_id, var, period)
    plt.savefig(os.path.join(fig_path, fig_name))
    plt.close()

# ============================================================
# Main SAM EOF Index Driver
# ============================================================
def run_sam_eof_index_generation(fig_path, out_path, mip, exp, relm, case_id, period, case_dict, sam_region, l_check_sam_region):
    for key in case_dict:
        case = key
        var = case_dict[key].var
        data = case_dict[key].path
        
    print("working on EOF SAM index generation for:", case, var)
    ds = open_dataset(data)
    ymds = '{}-{}-01'.format(period.split("-")[0][0:4], period.split("-")[0][4:6])
    ymde = '{}-{}-31'.format(period.split("-")[1][0:4], period.split("-")[1][4:6])
    ds = ds.sel(time=slice(ymds, ymde))
    
    sam_df, colnams = define_sam_eof(mip, exp, relm, case, case_id, period, "PSL", "hPa", ds, out_path)
    draw_sam_ts(fig_path, sam_df, colnams, sam_region, mip, exp, relm, case, case_id, period, "PSL", "hPa")
    return sam_df

# ============================================================
# Main SAM PSL Index Driver
# ============================================================
def run_sam_psl_index_generation(fig_path, out_path, mip, exp, relm, case_id, period, case_dict, sam_region, l_check_sam_region):
    for key in case_dict:
        case = key
        var = case_dict[key].var
        data = case_dict[key].path
        
    print("working on PSL SAM index generation for:", case, var)
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
        
    sam_df, colnams = define_sam_psl(mip, exp, relm, case, case_id, period, "PSL", "hPa", da, out_path)
    draw_sam_ts(fig_path, sam_df, colnams, sam_region, mip, exp, relm, case, case_id, period, "PSL", "hPa")
    return sam_df

# ============================================================
# Helper: Save lead-lag NetCDF for SAM
# ============================================================
def save_sam_leadlag_nc(corll, regll, pvlll, leadlag, mip, exp, relm, case, case_id, period, var, vunt, reg_idx, out_path):
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
# Main SAM Lead-Lag Driver
# ============================================================
def sam_lead_lag_anl(mip, exp, relm, case, case_id, period, out_path, fig_path, da, sam, reg_idx, time_vals, var):
    lons = da['longitude'][:]
    lats = da['latitude'][:]
    clm = da.groupby('time.month').mean(dim='time')
    anm = (da.groupby('time.month') - clm)
    vstr = da.name
    vunt = da.units
    samSD = sam / sam.std(dim='time')
    
    rsamSD = samSD.copy()
    ranm = anm.copy()
    rsamSD[:] = low_pass(1.0 / 5.0, samSD, axis=0)
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
        cor = xr.corr(rsamSD, rdanm.shift(time=tllg), dim="time")
        reg = xr.cov(rsamSD, rdanm.shift(time=tllg), dim="time") / rsamSD.var(dim='time', skipna=True).values
        pvl = pearson_r_p_value(rsamSD, rdanm.shift(time=tllg), dim="time")
        corll.append(cor)
        regll.append(reg)
        pvlll.append(pvl)
        
    save_sam_leadlag_nc(corll, regll, pvlll, leadlag, mip, exp, relm, case, case_id, period, var, vunt, reg_idx, out_path)
    
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
    
    sam_region = {'west': -180., 'east': 180., 'south': -90., 'north': -20.}
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
            ax, fill = draw_regression_map(vstr, vunt, lons, lats, corll[k], regll[k], pvlll[k], fig, sam_region,
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
    
    fig.suptitle('Corr. & Regress with SAM{}: {}({})'.format(reg_idx, vstr, vunt), fontsize=fontsize, y=0.9)
    plt.rcParams["font.family"] = "sans-serif"
    plt.draw()
    
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    fig_name = "fig_{}_leadlag_map_{}_{}_{}_{}_{}_{}_{}.pdf".format(reg_idx, mip, exp, case, relm, case_id, var, period)
    plt.savefig(os.path.join(fig_path, fig_name))
    plt.close()

def run_sam_leadlag_analysis(fig_path, out_path, mip, exp, relm, case_id, period, case_dict, sam_region, reg_idx, l_check_sam_region):
    for key in case_dict:
        case = key
        var = case_dict[key].var
        data = case_dict[key].path
        
    print("working on SAM lead-lag for:", case, var)
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
        
    sam_index_file = os.path.join(out_path.replace("lead_lag", "raw_index"), 
                                  "{}.{}.{}.{}.{}.{}.{}.csv".format(mip, exp, case, relm, case_id, "PSL", period))
    sam_df = pd.read_csv(sam_index_file)
    sam_exist = False
    for col in sam_df.columns:
        if reg_idx in col:
            sam = sam_df[col].to_xarray()
            sam_exist = True
    if not sam_exist:
        raise ValueError('Index {} not exist, please check....'.format(reg_idx))
        
    sam = sam.assign_coords({"time": da.time})
    sam_lead_lag_anl(mip, exp, relm, case, case_id, period, out_path, fig_path, da, sam, reg_idx, da.time, var)
