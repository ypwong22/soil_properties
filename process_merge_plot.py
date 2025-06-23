import xarray as xr
import matplotlib.pyplot as plt
import os
import cartopy.crs as ccrs

for region in ['conus','na']:
    filename = os.path.join(os.environ['SHARDIR'], 'Soil_Properties', 'intermediate', 
                            f'merged_properties_{region}.nc')

    ds = xr.open_dataset(filename)

    # Convenient list of names that usually hold latitude / longitude
    POSSIBLE_LAT = {"lat", "latitude", "y"}
    POSSIBLE_LON = {"lon", "longitude", "x"}

    for name, da in ds.data_vars.items():
        # --- Skip variables that don’t have a lat–lon grid ---------------------
        if not (set(da.dims) & POSSIBLE_LAT and set(da.dims) & POSSIBLE_LON):
            continue

        # --- Collapse a leading level/time/height/etc. if present --------------
        if da.ndim == 3:
            da = da.mean(dim=da.dims[0])

        # --- Identify the actual lat & lon dimension names ---------------------
        lat_dim = next(dim for dim in da.dims if dim in POSSIBLE_LAT)
        lon_dim = next(dim for dim in da.dims if dim in POSSIBLE_LON)

        fig = plt.figure(figsize=(8, 5))
        ax = plt.axes(projection=ccrs.PlateCarree())

        # pcolormesh treats boundaries correctly and is faster than contourf
        im = da.plot.pcolormesh(
            ax=ax,
            x=lon_dim,
            y=lat_dim,
            cmap="viridis",
            add_colorbar=False,
            transform=ccrs.PlateCarree(),
        )

        ax.coastlines()
        ax.set_title(name)
        cbar = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, aspect=50)
        cbar.set_label(da.attrs.get("units", ""))
        plt.tight_layout()
        plt.savefig(os.path.join(os.environ['SHARDIR'],'Soil_Properties', 'intermediate',
                                f'merged_properties_{region}_{name}.png'), dpi = 600., 
                    bbox_inches = 'tight')
        plt.close(fig)

    ds.close()