#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 19:41:33 2025

@author: frederikmobiusrygaard
"""

#%% Modules

import numpy as np

from itertools import chain

from numpy.typing import ArrayLike

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

import cartopy.crs as ccrs
import cartopy.feature as cfeature

#%% Helper function

def draw_map(m, scale=0.2):
    # draw a shaded-relief image
    m.shadedrelief(scale=scale)
    
    # lats and longs are returned as a dictionary
    lats = m.drawparallels(np.linspace(-90, 90, 13))
    lons = m.drawmeridians(np.linspace(-180, 180, 13))

    # keys contain the plt.Line2D instances
    lat_lines = chain(*(tup[1][0] for tup in lats.items()))
    lon_lines = chain(*(tup[1][0] for tup in lons.items()))
    all_lines = chain(lat_lines, lon_lines)
    
    # cycle through these lines and set the desired style
    for line in all_lines:
        line.set(linestyle='-', alpha=0.3, color='w')
        
def geodetic_to_unit_vector(lon_deg: float, lat_deg: float) -> np.ndarray:
    lat_rad = np.deg2rad(lat_deg)
    lon_rad = np.deg2rad(lon_deg)
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    return np.array([x, y, z])

def unit_vector_to_geodetic(vec: np.ndarray) -> tuple[float, float]:
    x, y, z = vec / np.linalg.norm(vec)
    lat_rad = np.arcsin(z)
    lon_rad = np.arctan2(y, x)
    return np.rad2deg(lon_rad), np.rad2deg(lat_rad)


#%% Plot of the Earth in chart

def plot_earth_chart(start_coordinate:ArrayLike,
                     end_coordinate:ArrayLike,
                     flight_path:ArrayLike,
                     figure_path:str='',
                     )->None:
    
    fig = plt.figure(figsize=(8, 6), edgecolor='w')
    m = Basemap(projection='moll', resolution=None,
                lat_0=0, lon_0=0)
    xp, yp = m(start_coordinate[0], start_coordinate[1])
    xc, yc = m(end_coordinate[0], end_coordinate[1])

    xfp, yfp = m(flight_path.T[0], flight_path.T[1])


    x_flightpath, y_flightpath = m(flight_path[0], flight_path[1])
    m.scatter(xp, yp, marker='o', color='black', s=10)
    m.scatter(xc, yc, marker='o', color='black', s=10)
    m.plot(xfp, yfp, color="red", linewidth=1.0)

    draw_map(m)

    # Get the current Axes object
    ax = plt.gca()
    
    fig.tight_layout()
    
    save_path = f'{figure_path}earth_chart.png'
    fig.savefig(save_path, format='png', pad_inches=0.1, bbox_inches='tight')
    
    plt.close()
    
    return

#%% Plot region of earth chart

def plot_earth_region_chart(start_coordinate: ArrayLike,
                            end_coordinate: ArrayLike,
                            flight_path: ArrayLike,
                            region_deviation: float = 10.,
                            figure_path: str = '',
                            ) -> None:
    """
    Plot flight path on a regional map.
    
    region: (lon_min, lon_max, lat_min, lat_max)
    """
    
    lon_min = np.min(flight_path[:, 0]) - region_deviation
    lon_max = np.max(flight_path[:, 0]) + region_deviation
    lat_min = np.min(flight_path[:, 1]) - region_deviation
    lat_max = np.max(flight_path[:, 1]) + region_deviation

    fig = plt.figure(figsize=(10, 8), edgecolor='w')
    m = Basemap(projection='cyl',
                resolution='l',
                llcrnrlon=lon_min, urcrnrlon=lon_max,
                llcrnrlat=lat_min, urcrnrlat=lat_max)

    # Convert coordinates
    xp, yp = m(start_coordinate[0], start_coordinate[1])
    xc, yc = m(end_coordinate[0], end_coordinate[1])
    xfp, yfp = m(flight_path[:, 0], flight_path[:, 1])

    # Plot flight path and points
    m.drawcoastlines()
    m.drawcountries()
    m.drawmapboundary(fill_color='lightblue')
    m.fillcontinents(color='tan', lake_color='lightblue')

    m.scatter(xp, yp, marker='o', color='black', s=50)
    m.scatter(xc, yc, marker='o', color='black', s=50)
    m.plot(xfp, yfp, color="red", linewidth=2.0)

    draw_map(m)
    
    fig.tight_layout()
    
    save_path = f'{figure_path}earth_region_chart.png'
    fig.savefig(save_path, format='png', pad_inches=0.1, bbox_inches='tight')
    
    plt.close()


#%% Plot 3D earth

def plot_3d_earth(start_coordinate: ArrayLike,
                  end_coordinate: ArrayLike,
                  flight_path: ArrayLike,
                  globe_scale: float = 0.5,
                  path_color: str = "red",
                  path_width: float = 2.0,
                  marker_size: int = 20,
                  figure_path: str = '',
                  start_city: str = '',
                  end_city: str = '',
                  ) -> None:
    """
    Plot a 3D globe centered on the great-circle view of the flight path.
    """
    # Convert to 3D vectors
    vec_start = geodetic_to_unit_vector(*start_coordinate)
    vec_end = geodetic_to_unit_vector(*end_coordinate)

    # Midpoint of great circle (normalize for unit sphere)
    vec_mid = (vec_start + vec_end) / 2
    vec_mid /= np.linalg.norm(vec_mid)

    # Convert back to lat/lon for projection center
    lon_0, lat_0 = unit_vector_to_geodetic(vec_mid)

    fig = plt.figure(figsize=(8, 8))
    m = Basemap(projection='ortho',
                resolution='l',
                lat_0=lat_0,
                lon_0=lon_0)

    # Convert coordinates to projection space
    xp, yp = m(start_coordinate[0], start_coordinate[1])
    xc, yc = m(end_coordinate[0], end_coordinate[1])
    xfp, yfp = m(flight_path[:, 0], flight_path[:, 1])

    # Background and flight path
    m.bluemarble(scale=globe_scale)
    m.plot(xfp, yfp, color=path_color, linewidth=path_width)
    m.scatter(xp, yp, marker='o', color='black', s=marker_size)
    m.scatter(xc, yc, marker='o', color='black', s=marker_size)

    # Add city name labels near start and end
    if start_city:
        plt.text(xp, yp, start_city, color='cyan', fontsize=10, fontweight='bold')
    if end_city:
        plt.text(xc, yc, end_city, color='cyan', fontsize=10, fontweight='bold')

    fig.tight_layout()

    save_path = f'{figure_path}earth_3d.png'
    fig.savefig(save_path, format='png', pad_inches=0.1, bbox_inches='tight')

    plt.close()

    
#%% Plot jet streams

def plot_jet_streams(lons:ArrayLike, 
                     lats:ArrayLike, 
                     u:ArrayLike, 
                     v:ArrayLike,
                     speed:ArrayLike,
                     flight_path:ArrayLike,
                     arrow_step=20,
                     region_deviation:float=10.,
                     title="Interpolated Jet Stream (250 hPa)",
                     figure_path:str='',
                     ):
    if lats[0] > lats[-1]:
        lats, u, v, speed = lats[::-1], u[::-1], v[::-1], speed[::-1]
        
    lon_min = np.min(flight_path[:,0]) - region_deviation
    lon_max = np.max(flight_path[:,0]) + region_deviation
    lat_min = np.min(flight_path[:,1]) - region_deviation
    lat_max = np.max(flight_path[:,1]) + region_deviation
    
    region_extent = [lon_min, lon_max, lat_min, lat_max]

    lon_s = np.array(lons)
    lat_s = np.array(lats)
    u_s = np.array(u) / 3.6 # convert to m/s
    v_s = np.array(v) / 3.6 # convert to m/s
    speed_s = np.array(speed) / 3.6 # convert to m/s

    lon_grid, lat_grid = np.meshgrid(lon_s, lat_s)

    # Optional region cropping
    if region_extent:
        lon_min, lon_max, lat_min, lat_max = region_extent
        mask_lon = (lon_s >= lon_min) & (lon_s <= lon_max)
        mask_lat = (lat_s >= lat_min) & (lat_s <= lat_max)

        lon_s = lon_s[mask_lon]
        lat_s = lat_s[mask_lat]
        speed_s = speed_s[np.ix_(mask_lat, mask_lon)]
        u_s = u_s[np.ix_(mask_lat, mask_lon)]
        v_s = v_s[np.ix_(mask_lat, mask_lon)]
        lon_grid, lat_grid = np.meshgrid(lon_s, lat_s)

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
    gl.top_labels = gl.right_labels = False

    if region_extent:
        ax.set_extent(region_extent, crs=ccrs.PlateCarree())

    cs = ax.contourf(lon_grid, lat_grid, speed_s, 60, cmap='coolwarm', transform=ccrs.PlateCarree())
    fig.colorbar(cs, orientation='horizontal', pad=0.05, label='Wind Speed (m/s)')

    ax.quiver(lon_grid[::arrow_step, ::arrow_step],
              lat_grid[::arrow_step, ::arrow_step],
              u_s[::arrow_step, ::arrow_step],
              v_s[::arrow_step, ::arrow_step],
              transform=ccrs.PlateCarree(),
              scale=700, width=0.002, headlength=4)

    # Plot flight path
    ax.plot(flight_path[:, 0], flight_path[:, 1], 
            marker='o', color='black', linestyle='-', linewidth=2, markersize=5,
            transform=ccrs.PlateCarree(), label='Flight Path')

    ax.legend(loc='upper right')
    plt.title(title)
    
    fig.tight_layout()
    
    save_path = f'{figure_path}earth_region_chart.png'
    fig.savefig(save_path, format='png', pad_inches=0.1, bbox_inches='tight')
    
    plt.close()
