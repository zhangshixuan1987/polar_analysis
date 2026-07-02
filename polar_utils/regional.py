import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from polar_utils.common import Case, low_pass, detrend_dim, slice_region, pearson_r_p_value, draw_regression_map, open_dataset

# ============================================================
# Regional Mean Climatology & Indices
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

def define_rgmn(da, region):
    times = da.time.dt.strftime("%Y-%m-%d")
    time_str = da.time.dt.strftime("%Y-%m-%d")
    ntime = len(times)
    print("number of total months in regional data: ", ntime)

    clm = da.groupby('time.month').mean(dim='time')
    anm = (da.groupby('time.month') - clm)
    dclm = anm.copy()
    for i in range(len(clm)):
        dclm[i::12, :, :] = clm[i, :, :].copy()

    rgmn_val = wgt_areaave(da, region)
    rgmn_clm = wgt_areaave(dclm, region)
    rgmn_ind = wgt_areaave(anm, region)
    rgmnSD_ind = rgmn_ind / rgmn_ind.std(dim='time')
    
    rrgmn_ind = low_pass(1.0 / 5.0, rgmn_ind, axis=0)
    orgmn_ind = low_pass(1.0 / 3.0, rgmn_ind, axis=0)
    rrgmnSD_ind = low_pass(1.0 / 5.0, rgmnSD_ind, axis=0)
    orgmnSD_ind = low_pass(1.0 / 3.0, rgmnSD_ind, axis=0)

    df = pd.DataFrame()
    df['time'] = time_str
    df['rgmn_val'] = rgmn_val
    df['rgmn_clm'] = rgmn_clm
    df['rgmn_idx'] = rgmn_ind
    df['rrgmn_idx'] = rrgmn_ind
    df['orgmn_idx'] = orgmn_ind
    df['rgmnSD_idx'] = rgmnSD_ind
    df['rrgmnSD_idx'] = rrgmnSD_ind
    df['orgmnSD_idx'] = orgmnSD_ind
    
    df = df[['time', 'rgmn_val', 'rgmn_clm', 'rgmn_idx', 'rrgmn_idx', 'orgmn_idx', 'rgmnSD_idx', 'rrgmnSD_idx', 'orgmnSD_idx']]
    return df.reset_index(drop=True)

def draw_rgmn_ts(fig_path, rgmn_df, region_name, mip, exp, case, relm, case_id, var, vunt, period):
    time = rgmn_df['time']
    xtime = np.linspace(1, len(time), len(time))
    years = int(time[0].split("-")[0])

    xtick = np.arange(0, len(time))
    xlabs = np.arange(0, len(time)) / 12.0 + years
    fontsize = 16
    fig = plt.figure(figsize=(8, 11))
    
    plot_cols = ['rgmn_val', 'rgmn_clm', 'rgmn_idx', 'rrgmn_idx', 'rgmnSD_idx', 'rrgmnSD_idx']
    for i, col in enumerate(plot_cols):
        var0 = np.array(rgmn_df[col])
        var1 = low_pass(1.0 / 11.0, var0, axis=0)
        ax = fig.add_subplot(len(plot_cols), 1, i + 1)
        ax.plot(xtime, var0, color='grey', alpha=1.0, linewidth=0.8, label='monthly')
        ax.plot(xtime, var1, color='black', alpha=1.0, linewidth=1.4, label='11-point Hamming')
        ax.set_xlabel('Time (years)')
        ax.set_ylabel(col)
        ax.set_xlim(0, len(time))
        ax.set_xticks(xtick[::120])
        ax.set_xticklabels(xlabs[::120].astype(int))
        ax.grid(True)
        if i + 1 == len(plot_cols):
            ax.legend(loc='lower right', prop={'size': 8})
            
    plt.suptitle('Regional {} indices ({}, {})'.format(region_name, var, vunt), fontsize=fontsize * 1.1)
    plt.tight_layout()
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    fig_name = "fig_ts_{}_{}_{}_{}_{}_{}_{}_{}.pdf".format(region_name, mip, exp, case, relm, case_id, var, period)
    plt.savefig(os.path.join(fig_path, fig_name))
    plt.close()

def run_regional_mean_generation(fig_path, out_path, mip, exp, relm, period, case_id, case_dict, region_dict, l_check_region):
    for key in case_dict:
        case = key
        var = case_dict[key].var
        data = case_dict[key].path
        
    print("working on regional mean generation for:", case, var)
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
        
    for rname, rbounds in region_dict.items():
        rgmn_df = define_rgmn(da, rbounds)
        draw_rgmn_ts(fig_path, rgmn_df, rname, mip, exp, case, relm, case_id, var, da.units, period)
        
        # Save output csv
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        out_file = '{}.{}.{}.{}.{}.{}.{}.csv'.format(rname, mip, exp, case, case_id, var, period)
        rgmn_df.to_csv(os.path.join(out_path, out_file), index=False)

# ============================================================
# MPAS Regional Diagnostics
# ============================================================
def plot_mpas_ts(fig_path, df, var, mip, exp, case, case_id, period, regions):
    time = df['time']
    xtime = np.linspace(1, len(time), len(time))
    years = int(time[0].split("-")[0])

    xtick = np.arange(0, len(time))
    xlabs = np.arange(0, len(time)) / 12.0 + years
    fontsize = 16
    fig = plt.figure(figsize=(10, 15))
    
    # plot up to 8 regions
    for i, reg in enumerate(regions[:8]):
        var0 = np.array(df[reg])
        var1 = low_pass(1.0 / 11.0, var0, axis=0)
        ax = fig.add_subplot(min(len(regions), 8), 1, i + 1)
        ax.plot(xtime, var0, color='grey', alpha=1.0, linewidth=0.8, label='monthly')
        ax.plot(xtime, var1, color='black', alpha=1.0, linewidth=1.4, label='11-point Hamming')
        ax.set_xlabel('Time (years)')
        ax.set_ylabel(reg)
        ax.set_xlim(0, len(time))
        ax.set_xticks(xtick[::120])
        ax.set_xticklabels(xlabs[::120].astype(int))
        ax.grid(True)
        if i + 1 == min(len(regions), 8):
            ax.legend(loc='lower right', prop={'size': 8})
            
    plt.suptitle('MPAS {} indices ({})'.format(var, period), fontsize=fontsize * 1.1)
    plt.tight_layout()
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    fig_name = "fig_ts_{}_{}_{}_{}_{}_{}.pdf".format(mip, exp, case, case_id, var, period)
    plt.savefig(os.path.join(fig_path, fig_name))
    plt.close()

def run_mpas_regional_generation(fig_path, out_path, mip, exp, relm, period, case_id, case_dict):
    case = list(case_dict.keys())[0]
    var_name = case_dict[case].var
    ftpl = case_dict[case].path
    
    if len(ftpl) > 1:
        for i, ff in enumerate(ftpl):
            if i == 0:
                ds = open_dataset(ff)
            else:
                ds = xr.merge([ds, open_dataset(ff)], compat='override')
    else:
        ds = open_dataset(ftpl[0])

    times = []
    for yy, mm in zip(ds['year'], ds['month']):
        times.append('{:04d}-{:02d}-15'.format(yy, mm))
    ds['Time'] = times 
    
    ymds = '{}-{}-01'.format(period.split("-")[0][0:4], period.split("-")[0][4:6])
    ymde = '{}-{}-31'.format(period.split("-")[1][0:4], period.split("-")[1][4:6])
    da = ds.sel(Time=slice(ymds, ymde))
    
    title = ['time', 'year', 'month']
    regnams = da['regionNames'].data
    regstrs = [reg.replace(" ", "_") for reg in regnams]
    title.extend(regstrs)

    var_list = []
    for var in list(da.keys()):
        if var != "zbounds" and len(da[var].shape) > 1:
            outdata = pd.DataFrame([], columns=title)
            outdata['time'] = da['Time'].data 
            outdata['year'] = da['year'].data
            outdata['month'] = da['month'].data
            for i, reg in enumerate(regstrs): 
                outdata[reg] = da[var][i, :].data
            var_list.append(var)
            
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            out_file = '{}.{}.{}.{}.{}.{}.csv'.format(mip, exp, case, case_id, var, period)
            outdata.to_csv(os.path.join(out_path, out_file), index=False)
            
    for var in var_list:
        ff = os.path.join(out_path, '{}.{}.{}.{}.{}.{}.csv'.format(mip, exp, case, case_id, var, period)) 
        df = pd.read_csv(ff, index_col=False)
        plot_mpas_ts(fig_path, df, var, mip, exp, case, case_id, period, regstrs)
