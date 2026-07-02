import os
import glob
import collections
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from skimage.feature import peak_local_max
import regionmask

from polar_utils.common import (
    Case,
    low_pass,
    detrend_dim,
    slice_region,
    pearson_r_p_value,
    draw_regression_map,
    draw_regional_box,
    plot_regions_mask
)

# ============================================================
# ASL sector mean
# ============================================================
def asl_sector_mean(da, region):
    latn = region['north']
    lats = region['south']
    lone = region['east']
    lonw = region['west']
    
    lat_name = 'latitude' if 'latitude' in da.dims else ('lat' if 'lat' in da.dims else None)
    lon_name = 'longitude' if 'longitude' in da.dims else ('lon' if 'lon' in da.dims else None)
    
    lat_slice = slice(lats, latn) if da[lat_name][0] < da[lat_name][-1] else slice(latn, lats)
    lon_slice = slice(lonw, lone) if da[lon_name][0] < da[lon_name][-1] else slice(lone, lonw)
    
    a = da.sel(**{lat_name: lat_slice, lon_name: lon_slice}).mean().values
    return a

# ============================================================
# Find pressure lows (get_lows)
# ============================================================
def get_lows(da, asl_region, min_dist, num_peak, exclue_border):
    lons_name = 'longitude' if 'longitude' in da.dims else 'lon'
    lats_name = 'latitude' if 'latitude' in da.dims else 'lat'
    lons = da[lons_name].values
    lats = da[lats_name].values
    
    sector_mean_pres = asl_sector_mean(da, asl_region)
    threshold = sector_mean_pres
    time_str = str(da.time.values)[:10]
    
    da_max = da.max().values
    da = da.fillna(da_max)
    invert_data = (da * -1.).values
    
    if threshold is None:
        threshold_abs = invert_data.mean()
    else:
        threshold_abs = threshold * -1
        
    minima_yx = peak_local_max(invert_data,
                               min_distance=min_dist,
                               num_peaks=num_peak,
                               exclude_border=exclue_border,
                               threshold_abs=threshold_abs)
    
    minima_lat, minima_lon, pressure = [], [], []
    for minima in minima_yx:
        minima_lat.append(lats[minima[0]])
        minima_lon.append(lons[minima[1]])
        pressure.append(da.values[minima[0], minima[1]])
        
    df = pd.DataFrame()
    df['lat']        = minima_lat
    df['lon']        = minima_lon
    df['ActCenPres'] = pressure
    df['SectorPres'] = sector_mean_pres
    df['time']       = time_str
    df['RelCenPres'] = df['ActCenPres'] - df['SectorPres']
    df = df[['time', 'lon', 'lat', 'ActCenPres', 'SectorPres', 'RelCenPres']]
    df = df.reset_index(drop=True)
    return df

# ============================================================
# Define ASL index dataframe
# ============================================================
def define_asl(times, df, region, l_allow_no_asl, mip, exp, relm, case, case_id, period, vstr, out_path):
    df2 = df[(df['lon'] > region['west'])  & 
             (df['lon'] < region['east'])  & 
             (df['lat'] > region['south']) & 
             (df['lat'] < region['north']) ]
    
    df2 = df2.loc[df2.groupby('time')['ActCenPres'].idxmin()]
    df2 = df2.reset_index(drop=True)
    
    if len(df2) != len(times):
        c = list(set(df2['time']).symmetric_difference(times.values))
        if l_allow_no_asl:
            for time_str in c:
                asl_df               = pd.DataFrame()
                asl_df['lat']        = [np.nan]
                asl_df['lon']        = [np.nan]
                asl_df['ActCenPres'] = [np.nan]
                asl_df['SectorPres'] = [np.nan]
                asl_df['RelCenPres'] = [np.nan]
                asl_df['time']       = time_str
                df2 = pd.concat([df2, asl_df], ignore_index=True)
        else:
            df1 = df.loc[df.groupby('time')['ActCenPres'].idxmin()]
            df1 = df1.reset_index(drop=True)
            for time_str in c:
                indx = list(df1['time']).index(time_str)
                asl_df               = pd.DataFrame()
                asl_df['lat']        = [df1['lat'][indx]] 
                asl_df['lon']        = [df1['lon'][indx]]
                asl_df['ActCenPres'] = [df1['ActCenPres'][indx]]
                asl_df['SectorPres'] = [df1['SectorPres'][indx]]
                asl_df['RelCenPres'] = [df1['RelCenPres'][indx]]
                asl_df['time']       = time_str
                df2 = pd.concat([df2, asl_df], sort=True)
                
    colnams = ['time', 'lon', 'lat', 'ActCenPres', 'SectorPres', 'RelCenPres']
    df2 = df2.sort_values(by="time")
    df2 = df2.reset_index(drop=True)
    df2 = df2[colnams]
    
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    out_file = '{}.{}.{}.{}.{}.{}.{}.csv'.format(mip, exp, case, relm, case_id, vstr, period)
    df2.to_csv(os.path.join(out_path, out_file), index=False)
    return df2, colnams

# ============================================================
# Plot ASL location map
# ============================================================
def draw_asl_loc(da, mask, asl_df, asl_region, mip, exp, relm, case, case_id, period, vstr, vunt, fig_path):
    da_mask = da.where(mask == 0)
    da_mask = slice_region(da_mask, asl_region)
    
    lat_name = 'latitude' if 'latitude' in da.dims else 'lat'
    lon_name = 'longitude' if 'longitude' in da.dims else 'lon'
    
    plt.figure(figsize=(20, 15))
    for i in range(0, min(12, len(da_mask.time))):
        da_2D = da_mask.isel(time=i)
        da_2D = da_2D.sel(**{lat_name: slice(-90, -55), lon_name: slice(165, 305)})
        
        ax = plt.subplot(3, 4, i + 1,
                         projection=ccrs.Stereographic(central_longitude=0., central_latitude=-90.))
        ax.set_extent([165, 305, -85, -55], ccrs.PlateCarree())
        
        da_2D.plot.contourf(lon_name, lat_name, cmap='Reds',
                            transform=ccrs.PlateCarree(),
                            add_colorbar=False,
                            levels=np.linspace(np.nanmin(da_2D.values), np.nanmax(da_2D.values), 20))
        
        ax.set_title('{}({}): {}'.format(vstr, vunt, str(da_2D.time.values)[0:7]))
        df2 = asl_df[asl_df['time'] == str(da_2D.time.values)[0:10]]
        if len(df2) > 0:
            ax.plot(df2['lon'], df2['lat'], 'mx', transform=ccrs.PlateCarree())
        draw_regional_box(asl_region)
        
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    fig_name = "fig_aslloc_map_{}_{}_{}_{}_{}_{}_1-12month.pdf".format(mip, exp, case, relm, case_id, vstr)
    plt.savefig(os.path.join(fig_path, fig_name))
    plt.close()

# ============================================================
# Plot ASL time series
# ============================================================
def draw_asl_ts(fig_path, asl_df, colnams, asl_region, mip, exp, relm, case, case_id, period, var, vunt):
    time = asl_df['time']
    xtime = np.linspace(1, len(time), len(time))
    years = int(time[0].split("-")[0])
    
    xtick = np.arange(0, len(time))
    xlabs = np.arange(0, len(time)) / 12.0 + years
    fontsize = 16
    fig = plt.figure(figsize=(8, 11))
    for i, col in enumerate(colnams[1:]):
        var0 = np.array(asl_df[col])
        var1 = low_pass(1.0 / 11.0, var0, axis=0)
        ax = fig.add_subplot(len(colnams), 1, i + 1)
        ax.plot(xtime, var0, color='grey', alpha=1.0, linewidth=0.8, label='monthly')
        ax.plot(xtime, var1, color='black', alpha=1.0, linewidth=1.4, label='11-point Hamming')
        ax.set_xlabel('Time (years)')
        ax.set_ylabel(col)
        ax.set_xlim(0, len(time))
        ax.set_xticks(xtick[::120])
        ax.set_xticklabels(xlabs[::120].astype(int))
        ax.grid(True)
        if i + 1 == len(colnams):
            ax.legend(loc='lower right', prop={'size': 8})
            
    plt.suptitle('ASL indices ({},{})'.format(var, vunt), fontsize=fontsize * 1.1)
    plt.tight_layout()
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    fig_name = "fig_ts_{}_{}_{}_{}_{}_{}_{}.pdf".format(mip, exp, case, relm, case_id, var, period)
    plt.savefig(os.path.join(fig_path, fig_name))
    plt.close()

# ============================================================
# Helper: Save regression data to NetCDF
# ============================================================
def save_regression_data(vcor, vreg, pval, vsig, mip, exp, relm, case, case_id, period, var, vunt, out_path):
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
# Plot ASL regression map
# ============================================================
def draw_asl_map(out_path, fig_path, da, asl_df, colnams, asl_region, mip, exp, relm, case, case_id, period, var, vunt):
    lons = da['longitude'][:]
    lats = da['latitude'][:]
    clm = da.groupby('time.month').mean(dim='time')
    anm = (da.groupby('time.month') - clm)
    vstr = da.name
    vunt = da.units
    
    time = asl_df['time']
    asl_df = asl_df.set_index('time')
    asl_exist = False
    for col in colnams:
        if 'RelCenPres' in col:
            asl = asl_df[col].to_xarray()
            asl_exist = True
    if not asl_exist:
        raise ValueError('RelCenPres not exist, please check....')
        
    asl = asl.assign_coords({"time": da.time})
    aslSD = asl / asl.std(dim='time')
    
    raslSD = aslSD.copy()
    ranm = anm.copy()
    raslSD[:] = low_pass(1.0 / 5.0, aslSD, axis=0)
    if np.isnan(anm).any():
        tmp1 = ranm.fillna(-99999)
        tmp1 = low_pass(1.0 / 5.0, tmp1, axis=0)
        ranm = anm.copy()
        ranm[:, :, :] = tmp1[:, :, :]
        ranm = ranm.where(ranm > -10000)
    else:
        ranm[:, :, :] = low_pass(1.0 / 5.0, anm[:, :, :], axis=0)
    rdanm = detrend_dim(ranm, 'time', 1)
    
    vcor = xr.corr(raslSD, rdanm, dim="time")
    vreg = xr.cov(raslSD, rdanm, dim="time") / raslSD.var(dim='time', skipna=True).values
    pval = pearson_r_p_value(raslSD, rdanm, dim="time")
    
    vsig = vreg.copy()
    vsig = vsig.where(pval <= 0.05)
    
    save_regression_data(vcor, vreg, pval, vsig, mip, exp, relm, case, case_id, period, var, vunt, out_path)
    
    vmin = -1.0
    vmax = 1.0
    nlev = 21
    fontsize = 16
    bartitle = 'Regressed {}({})'.format(vstr, vunt)
    fig = plt.figure(figsize=(10, 12))
    grid = fig.add_gridspec(ncols=1, nrows=1)
    ax1, fill1 = draw_regression_map(vstr, vunt, lons, lats, vcor, vreg, pval, fig, asl_region,
                                     fontsize, 'SAM-PSL Pattern', grid[0, 0], vmin, vmax, nlev)
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
# Main ASL Index Calculation Driver
# ============================================================
def run_asl_index_generation(fig_path, out_path, mip, exp, relm, case_id, period, case_dict, asl_region, 
                             asl_min_dist, asl_num_peak, asl_exc_bord, l_check_asl_region, l_allow_no_asl):
    for key in case_dict:
        if key == "mask":
            dmsk = case_dict[key].path
            vmsk = case_dict[key].var
        else:
            case = key
            var = case_dict[key].var
            data = case_dict[key].path
            
    print("working on ASL index generation for:", case, var)
    
    ds = xr.open_dataset(data)
    ymds = '{}-{}-01'.format(period.split("-")[0][0:4], period.split("-")[0][4:6])
    ymde = '{}-{}-31'.format(period.split("-")[1][0:4], period.split("-")[1][4:6])
    ds = ds.sel(time=slice(ymds, ymde))
    
    lat_name = 'latitude' if 'latitude' in ds.dims else ('lat' if 'lat' in ds.dims else None)
    lon_name = 'longitude' if 'longitude' in ds.dims else ('lon' if 'lon' in ds.dims else None)
    if lat_name and lon_name and (lat_name == 'lat' or lon_name == 'lon'):
        ds = ds.rename({lat_name: 'latitude', lon_name: 'longitude'})
        
    if os.path.exists(dmsk):
        dsm = xr.open_dataset(dmsk)
        vmsk_lat = 'latitude' if 'latitude' in dsm.dims else ('lat' if 'lat' in dsm.dims else dsm[vmsk].dims[0])
        vmsk_lon = 'longitude' if 'longitude' in dsm.dims else ('lon' if 'lon' in dsm.dims else dsm[vmsk].dims[1])
        if vmsk_lat == 'lat' or vmsk_lon == 'lon':
            dsm = dsm.rename({vmsk_lat: 'latitude', vmsk_lon: 'longitude'})
        mask = dsm[vmsk]
        mask = mask / 100.0
    else:
        print("Warning: land/sea mask not exist, derive it...")
        lons_tmp = ds.longitude.copy()
        lats_tmp = ds.latitude
        if lons_tmp.values.min() > -1:
            lons_tmp = (lons_tmp + 180) % 360 - 180
        land_110 = regionmask.defined_regions.natural_earth_v5_0_0.land_110
        mask_da  = land_110.mask(lons_tmp.values, lats_tmp.values)
        mask = (~np.isnan(mask_da.values)).astype(float)
        mask = xr.DataArray(mask, coords=[ds.latitude, ds.longitude], dims=['latitude', 'longitude'])
        
    da = ds[var]
    if da.units == "Pa":
        da = da / 100.
        da = da.assign_attrs(units='hPa')
    vstr = da.name.upper()
    vunt = da.units
    
    if l_check_asl_region:
        da_t = da.sel(time=da.time[0:11])
        da_mask2 = da_t.where(mask == 0)
        da_nmsk = slice_region(da_t, asl_region)
        da_mask2 = slice_region(da_mask2, asl_region)
        plot_regions_mask(fig_path, da_nmsk, da_mask2, case)
        
    times = da.time.dt.strftime("%Y-%m-%d")
    ntime = len(times)
    all_lows_dfs = pd.DataFrame()
    print("number of total months in data: ", ntime)
    for t in range(ntime):
        da_t = da.isel(time=t)
        da_mask2 = da_t.where(mask == 0)
        da_mask2 = slice_region(da_mask2, asl_region)
        all_lows_df = get_lows(da_mask2, asl_region, asl_min_dist, asl_num_peak, asl_exc_bord)
        all_lows_dfs = pd.concat([all_lows_dfs, all_lows_df], ignore_index=True)
        
    asl_df, colnams = define_asl(times, all_lows_dfs, asl_region, l_allow_no_asl, mip, exp, relm, case, case_id, period, vstr, out_path)
    
    draw_asl_loc(da, mask, asl_df, asl_region, mip, exp, relm, case, case_id, period, vstr, vunt, fig_path)
    draw_asl_ts(fig_path, asl_df, colnams, asl_region, mip, exp, relm, case, case_id, period, vstr, vunt)
    draw_asl_map(out_path, fig_path, ds[var], asl_df, colnams, asl_region, mip, exp, relm, case, case_id, period, vstr, vunt)
    return asl_df

# ============================================================
# ASL Lead-Lag Analysis Routines
# ============================================================
def save_leadlag_nc(corll, regll, pvlll, leadlag, mip, exp, relm, case, case_id, period, var, vunt, reg_idx, out_path):
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

def asl_lead_lag_anl(mip, exp, relm, case, case_id, period, out_path, fig_path, da, asl, reg_idx, time_vals, var):
    lons = da['longitude'][:]
    lats = da['latitude'][:]
    clm = da.groupby('time.month').mean(dim='time')
    anm = (da.groupby('time.month') - clm)
    vstr = da.name
    vunt = da.units
    aslSD = asl / asl.std(dim='time')
    
    raslSD = aslSD.copy()
    ranm = anm.copy()
    raslSD[:] = low_pass(1.0 / 5.0, aslSD, axis=0)
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
        cor = xr.corr(raslSD, rdanm.shift(time=tllg), dim="time")
        reg = xr.cov(raslSD, rdanm.shift(time=tllg), dim="time") / raslSD.var(dim='time', skipna=True).values
        pvl = pearson_r_p_value(raslSD, rdanm.shift(time=tllg), dim="time")
        corll.append(cor)
        regll.append(reg)
        pvlll.append(pvl)
        
    save_leadlag_nc(corll, regll, pvlll, leadlag, mip, exp, relm, case, case_id, period, var, vunt, reg_idx, out_path)
    
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
    
    asl_region = {'west': 170., 'east': 298., 'south': -80., 'north': -60.}
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
            ax, fill = draw_regression_map(vstr, vunt, lons, lats, corll[k], regll[k], pvlll[k], fig, asl_region,
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
    
    fig.suptitle('Corr. & Regress with ASL{}: {}({})'.format(reg_idx, vstr, vunt), fontsize=fontsize, y=0.9)
    plt.rcParams["font.family"] = "sans-serif"
    plt.draw()
    
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    fig_name = "fig_{}_leadlag_map_{}_{}_{}_{}_{}_{}_{}.pdf".format(reg_idx, mip, exp, case, relm, case_id, var, period)
    plt.savefig(os.path.join(fig_path, fig_name))
    plt.close()

def run_asl_leadlag_analysis(fig_path, out_path, mip, exp, relm, case_id, period, case_dict, asl_region, reg_idx, l_check_asl_region):
    for key in case_dict:
        case = key
        var = case_dict[key].var
        data = case_dict[key].path
        
    print("working on ASL lead-lag for:", case, var)
    
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
        
    asl_index_file = os.path.join(out_path.replace("lead_lag", "raw_index"), 
                                  "{}.{}.{}.{}.{}.{}.{}.csv".format(mip, exp, case, relm, case_id, "PSL", period))
    asl_df = pd.read_csv(asl_index_file)
    asl_exist = False
    for col in asl_df.columns:
        if reg_idx in col:
            asl = asl_df[col].to_xarray()
            asl_exist = True
    if not asl_exist:
        raise ValueError('Index {} not exist, please check....'.format(reg_idx))
        
    asl = asl.assign_coords({"time": da.time})
    asl_lead_lag_anl(mip, exp, relm, case, case_id, period, out_path, fig_path, da, asl, reg_idx, da.time, var)

# ============================================================
# ASL Climatology Index calculation (Step 1)
# ============================================================
def load_metric_file_list_step1(path, mip, exp, ver): 
    file_dict = {}
    ftest = glob.glob(os.path.join(path, "{}.{}.{}.{}.{}.csv".format(mip, exp, "*", ver, "*")))
    for ff in ftest: 
        if os.path.exists(ff):
            fname = ff.split("/")[-1]
            product = fname.split(".")[2]
            relm = fname.split(".")[3]
            if mip == "obs":
                product = product.replace("_", "-")
            if product not in file_dict.keys():
                file_dict[product] = {}
            if relm not in file_dict[product].keys():
                file_dict[product][relm] = ff 
    return file_dict 

def load_and_process_data_step1(file_dict, seasons, indices, out_path):
    for mip in file_dict.keys(): 
        for exp in file_dict[mip].keys():
            for prod in file_dict[mip][exp].keys():
                for relm in file_dict[mip][exp][prod].keys():
                    csvObj = file_dict[mip][exp][prod][relm]
                    data = pd.read_csv(csvObj)
                    data = data.reset_index(names=['year'])
                    data['year'] = pd.DatetimeIndex(data['time']).year
                    data['month'] = pd.DatetimeIndex(data['time']).month
                    data['day'] = pd.DatetimeIndex(data['time']).day
                    data = data.drop(columns=['time'])
                    for sea in seasons: 
                        print("processing seasonal climatology step 1:", mip, exp, prod, relm, sea)  
                        if sea in ['DJF']: 
                            sub_data  = data[data['month'].isin([12, 1, 2])].copy()
                            datmn = sub_data.groupby(by=['year'], dropna=True).mean()
                        elif sea in ['JJA']:
                            sub_data  = data[data['month'].isin([6, 7, 8])].copy()
                            datmn = sub_data.groupby(by=['year'], dropna=True).mean()
                        elif sea in ['MAM']:
                            sub_data  = data[data['month'].isin([3, 4, 5])].copy()
                            datmn = sub_data.groupby(by=['year'], dropna=True).mean()
                        elif sea in ['SON']:
                            sub_data  = data[data['month'].isin([9, 10, 11])].copy()
                            datmn = sub_data.groupby(by=['year'], dropna=True).mean()
                        elif sea in ['ANN']:
                            sub_data  = data
                            datmn = sub_data.groupby(by=['year'], dropna=True).mean()
                        elif sea in ['AC']:
                            sub_data  = data
                            datmn = sub_data.groupby(by=['month'], dropna=True).mean()
                        elif sea in ['Monthly']:
                            sub_data  = data
                            datmn = data 
                        
                        out_dir = os.path.join(out_path, mip, exp)
                        if not os.path.exists(out_dir):
                            os.makedirs(out_dir)
                        period = '{:04d}-{:04d}'.format(min(data['year']), max(data['year'])) 
                        out_file = 'ASL.index.{}.{}.{}.{}.csv'.format(prod, relm, sea, period)
                        out_file = os.path.join(out_dir, out_file) 
                        if os.path.exists(out_file):
                            os.remove(out_file)
                            
                        if sea in ['Monthly']:
                            month = datmn['month']
                            day = datmn['day']
                            datmn = datmn.drop(columns=['month', 'day'])
                            outdata = pd.DataFrame(datmn)
                            outdata.insert(1, "month", month)
                            outdata.insert(2, "day", day)
                            outdata.to_csv(out_file, index=False)
                        elif sea in ['DJF', "JJA", "MAM", "SON", "ANN"]:
                            year = np.arange(min(data['year']), max(data['year']) + 1)
                            datmn = datmn.drop(columns=['month', 'day'])
                            outdata = pd.DataFrame(datmn)
                            outdata.insert(0, "year", year[:len(outdata)])
                            outdata.to_csv(out_file, index=False)
                        else: # AC
                            month = np.arange(1, 13, 1)
                            datmn = datmn.drop(columns=['year'])
                            outdata = pd.DataFrame(datmn)
                            outdata.insert(0, "month", month) 
                            outdata.to_csv(out_file, index=False) 

def run_asl_index_clim_step1(mips, exps, ver, seasons, indices, data_path, out_path):
    file_dict = {}
    for mip in mips: 
        for exp in exps: 
            if mip not in file_dict.keys():
                file_dict[mip] = {}
            if exp not in file_dict[mip].keys():  
                file_dict[mip][exp] = {}
            file_dict[mip][exp] = load_metric_file_list_step1(data_path, mip, exp, ver)
             
    load_and_process_data_step1(file_dict, seasons, indices, out_path)

# ============================================================
# ASL Climatology Index calculation (Step 2)
# ============================================================
def load_metric_file_list_step2(path, mip, exp, sea): 
    file_dict = {}
    ftest = glob.glob(os.path.join(path, mip, exp, "ASL.index.{}.{}.{}.csv".format("*", sea, "*")))
    for ff in ftest: 
        if os.path.exists(ff):
            fname = ff.split("/")[-1]
            product = fname.split(".")[2]
            relm = fname.split(".")[3]
            if product not in file_dict.keys():
                file_dict[product] = {}
            file_dict[product][relm] = ff 
    return file_dict 

def process_mean_climatology(ref_data, test_data, indices, sea, period):
    metrics = ['mean', 'mean_obs', 'std', 'std_obs', 'bias', 'std_xyt', 'rms_xyt']
    metric_lib = {}
    for met in metrics: 
        for var in indices:
            if met not in metric_lib.keys():
                metric_lib[met] = {}  
            if var not in metric_lib[met].keys():
                metric_lib[met][var] = {}
                
    syear = int(period.split("-")[0])
    eyear = int(period.split("-")[1])
    mask1 = ref_data['year'].between(syear, eyear) if 'year' in ref_data.columns else ref_data['month'].between(1, 12)
    x1 = ref_data[mask1]
    
    if 'year' in ref_data.columns:
        syear1 = min(x1['year'])
        eyear1 = max(x1['year'])
        mask2 = test_data['year'].between(syear1, eyear1)
        y1 = test_data[mask2]
    else:
        y1 = test_data
        
    if sea in ['AC']: 
        x = x1.groupby(by=['month'], dropna=True).mean() 
        y = y1.groupby(by=['month'], dropna=True).mean()
    else:
        x = x1
        y = y1 

    for var in indices: 
        x0 = x[var].to_numpy().astype(np.float32)
        y0 = y[var].to_numpy().astype(np.float32)
        df = y0 - x0 
        ds = df * df 
        metric_lib['mean'][var]     = np.nanmean(y0, axis=0) 
        metric_lib['mean_obs'][var] = np.nanmean(x0, axis=0)
        metric_lib['std'][var]      = np.nanstd(y0, axis=0) 
        metric_lib['std_obs'][var]  = np.nanstd(x0, axis=0) 
        metric_lib['bias'][var]     = np.nanmean(df, axis=0)
        metric_lib['std_xyt'][var]  = np.nanstd(df, axis=0)
        metric_lib['rms_xyt'][var]  = np.sqrt(np.nanmean(ds))
        
    return metric_lib

def run_asl_index_clim_step2(obs_sets, obs_mips, test_mips, test_exps, periods, seasons, indices, data_path, out_path):
    for ii, obs in enumerate(obs_sets): 
        for sea in seasons: 
            if sea == "AC":
                season_tag = "Monthly"
            else:
                season_tag = sea 
            ofils = glob.glob(os.path.join(data_path, obs_mips[ii], 'historical',
                                           "ASL.index.{}.{}.*.csv".format(obs, season_tag)))
            if len(ofils) < 1: 
                print("Warning: no observational file for {}, skipping ...".format(obs))
                continue
                
            out_lib = {}
            period = periods[ii]
            ref_data = pd.read_csv(ofils[0], index_col=False)
            for mip in test_mips:
                for exp in test_exps:
                    test_file = load_metric_file_list_step2(data_path, mip, exp, season_tag)
                    for prod in test_file.keys():
                        for relm in test_file[prod].keys():
                            test_data = pd.read_csv(test_file[prod][relm], index_col=False) 
                            metric_lib = process_mean_climatology(ref_data, test_data, indices, sea, period)
                            for metric in metric_lib.keys():
                                dtmp = [mip, exp, prod, relm, '{}_{}'.format(prod, relm)]
                                title = ['mip', 'exp', 'model', 'run', 'model_run']
                                for var in metric_lib[metric].keys():
                                    title.append(var)  
                                    dtmp.append(metric_lib[metric][var])
                                if metric not in out_lib.keys():
                                    out_lib[metric] = pd.DataFrame([dtmp], columns=title)
                                else:
                                    out_lib[metric] = pd.concat([out_lib[metric], pd.DataFrame([dtmp], columns=title)], ignore_index=True)
                                    
            out_dir = os.path.join(out_path, obs)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            for metric in out_lib.keys():
                out_file = 'ASL.{}.clim.{}.{}.csv'.format(metric, sea, period)
                out_file = os.path.join(out_dir, out_file)
                if os.path.exists(out_file):
                    os.remove(out_file)  
                outdata = out_lib[metric]
                outdata = outdata.drop(columns=['mip', 'exp'])
                outdata.to_csv(out_file, index=False)
