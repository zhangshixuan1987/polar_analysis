import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
from scipy import stats
from scipy.signal import butter, filtfilt

# ============================================================
# Helper: open_dataset supporting glob/list of files
# ============================================================
def open_dataset(data):
    if isinstance(data, str) and ("*" in data or "?" in data):
        return xr.open_mfdataset(data)
    elif isinstance(data, (list, tuple)):
        return xr.open_mfdataset(data)
    else:
        return xr.open_dataset(data)

# ============================================================
# Case class
# ============================================================
class Case:
    def __init__(self, path, var, color, label):
        self._path  = path
        self._color = color
        self._label = label
        self._var   = var

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
    def var(self):
        return self._var

# ============================================================
# Helper: Pearson r p-value (replaces xskillscore)
# ============================================================
def pearson_r_p_value(a, b, dim, skipna=True):
    """Two-tailed p-value of Pearson correlation over dimension `dim`."""
    n = a.sizes[dim]
    r = xr.corr(a, b, dim=dim)
    t_stat = r * np.sqrt((n - 2) / (1 - r**2))
    pval = 2 * stats.t.sf(np.abs(t_stat.values), df=n - 2)
    return xr.DataArray(pval, dims=r.dims, coords=r.coords)

# ============================================================
# Helper: Blue-Yellow-Red color list (replaces cmaps + geocat)
# ============================================================
def get_byr_colorlist(n=20):
    """Return a list of n colors approximating NCL's BlueYellowRed (white at center)."""
    cmap = plt.cm.RdYlBu_r  # built-in Blue->Yellow->Red diverging colormap
    colors = [list(cmap(i / (n - 1))) for i in range(n)]
    mid = n // 2
    colors[mid - 1] = [1., 1., 1., 1.]
    colors[mid    ] = [1., 1., 1., 1.]
    return colors

# ============================================================
# Low-pass filter
# ============================================================
def low_pass(cutoff_freq, data, order=5, axis=-1):
    Wn = cutoff_freq
    b, a = butter(order, Wn, btype='lowpass', analog=False)
    data_filt = filtfilt(b, a, data, axis=axis, method="gust")
    return data_filt

# ============================================================
# Detrend along a dimension
# ============================================================
def detrend_dim(da, dim, deg=1):
    p = da.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    return da - fit

# ============================================================
# Slice region
# ============================================================
def slice_region(da, region, boarder=8):
    latn = region['north'] + boarder
    lats = region['south'] - boarder
    lone = region['east']  + boarder
    lonw = region['west']  - boarder
    
    # Handle dimension names (latitude vs lat, longitude vs lon)
    lat_name = 'latitude' if 'latitude' in da.dims else ('lat' if 'lat' in da.dims else None)
    lon_name = 'longitude' if 'longitude' in da.dims else ('lon' if 'lon' in da.dims else None)
    
    if lat_name and lon_name:
        # Determine slice order based on lat/lon coordinates
        lat_slice = slice(lats, latn) if da[lat_name][0] < da[lat_name][-1] else slice(latn, lats)
        lon_slice = slice(lonw, lone) if da[lon_name][0] < da[lon_name][-1] else slice(lone, lonw)
        da = da.sel(**{lat_name: lat_slice, lon_name: lon_slice})
    return da

# ============================================================
# Draw regional box
# ============================================================
def draw_regional_box(region, transform=None):
    if transform is None:
        transform = ccrs.PlateCarree()
    latn = region['north']
    lats = region['south']
    lone = region['east']
    lonw = region['west']
    
    # draw box
    x = [lonw, lone, lone, lonw, lonw]
    y = [lats, lats, latn, latn, lats]
    plt.plot(x, y, color='black', linewidth=1.5, transform=transform)

# ============================================================
# Plot region sanity check
# ============================================================
def plot_regions_mask(fig_path, da, da_mask, group):
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    plt.figure(figsize=(10, 5))
    
    # Handle dimension names (latitude vs lat, longitude vs lon)
    lat_name = 'latitude' if 'latitude' in da.dims else ('lat' if 'lat' in da.dims else None)
    lon_name = 'longitude' if 'longitude' in da.dims else ('lon' if 'lon' in da.dims else None)
    
    ax1 = plt.subplot(121, projection=ccrs.Stereographic(central_longitude=0., central_latitude=-90.))
    ax1.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
    da.isel(time=-1).plot.pcolormesh(lon_name, lat_name, cmap='jet',
                                     transform=ccrs.PlateCarree(), add_colorbar=False)
    
    ax2 = plt.subplot(122, projection=ccrs.Stereographic(central_longitude=0., central_latitude=-90.))
    ax2.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
    da_mask.isel(time=-1).plot.pcolormesh(lon_name, lat_name, cmap='jet',
                                          transform=ccrs.PlateCarree(), add_colorbar=False)
    
    plt.savefig(os.path.join(fig_path, "asl_region_{}.pdf".format(group)))
    plt.close()
    return

# ============================================================
# Draw regression map (replaces geocat.viz.util + cmaps)
# ============================================================
def draw_regression_map(vstr, vunt, lats, lons, cor, reg, pval, fig, region,
                        fontsize, title, grid_space, vmin, vmax, nlev):
    sig    = pval.copy()
    sig[:] = 1.0 - sig[:]
    t90 = 0.94
    t95 = 0.95
    rlabel = '{}({})'.format(vstr, vunt)

    ax = fig.add_subplot(grid_space,
                         projection=ccrs.PlateCarree(central_longitude=210))
    ax.coastlines(linewidth=0.5, alpha=0.6)

    # Set axes limits and ticks
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())

    # Add lat/lon tick labels
    ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
    ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())

    # Set tick label size
    ax.tick_params(labelsize=fontsize * 0.90)

    # Build color list
    color_list = get_byr_colorlist(nlev - 1)

    kwargs = dict(
        vmin=vmin,
        vmax=vmax,
        levels=nlev,
        colors=color_list,
        add_colorbar=False,
        transform=ccrs.PlateCarree(),
    )
    
    # Handle dimension names (latitude vs lat, longitude vs lon)
    lat_name = 'latitude' if 'latitude' in cor.dims else ('lat' if 'lat' in cor.dims else 'lat')
    lon_name = 'longitude' if 'longitude' in cor.dims else ('lon' if 'lon' in cor.dims else 'lon')
    
    fillplot = cor.plot.contourf(ax=ax, x=lon_name, y=lat_name, **kwargs)

    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=1)
    ax.add_feature(cfeature.COASTLINE, edgecolor='gray', linewidth=0.5, zorder=1)

    sig.plot.contourf(ax=ax, x=lon_name, y=lat_name, levels=[-1*t95, -1*t90, t90, t95], colors='none',
                      hatches=[None, None, None, '..', '..'], extend='both',
                      add_colorbar=False, transform=ccrs.PlateCarree())

    delc = 0.2
    levels = np.arange(-3, 0, delc)
    levels = np.append(levels, np.arange(delc, 3, delc))
    rad = reg.plot.contour(ax=ax, x=lon_name, y=lat_name, colors='black', alpha=0.8, linewidths=1.0,
                           add_labels=False, levels=levels, transform=ccrs.PlateCarree())
    pe = [PathEffects.withStroke(linewidth=2.0, foreground="w")]
    if hasattr(rad, 'lines'):
        plt.setp(rad.lines, path_effects=pe)
    elif hasattr(rad, 'collections'):
        plt.setp(rad.collections, path_effects=pe)

    # Set titles and labels
    ax.set_title(title,  loc='left',  fontsize=fontsize * 0.95)
    ax.set_title(rlabel, loc='right', fontsize=fontsize * 0.95)
    ax.set_xlabel("")
    ax.set_ylabel("")

    draw_regional_box(region)
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    return ax, fillplot
