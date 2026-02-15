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
from mpl_toolkits.axes_grid1 import make_axes_locatable

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patheffects import withStroke

from typing import Dict

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
    
#%% Plot earth

import matplotlib.patheffects as path_effects

def plot_3d_earth_multi_paths(start_coordinate: np.ndarray,
                              end_coordinate: np.ndarray,
                              flight_paths: list,
                              globe_scale: float = 0.5,
                              path_colors: list = None,
                              path_width: float = 2.0,
                              marker_size: int = 20,
                              figure_path: str = '',
                              start_city: str = '',
                              end_city: str = '',
                              city_label_size: int = 10) -> None:
    """
    Plot a 3D globe with multiple flight paths between the same start and end points.
    
    City labels now have a white halo and are offset so they don't overlap the markers.
    """
    import numpy as np
    from mpl_toolkits.basemap import Basemap
    import matplotlib.pyplot as plt

    # Compute globe center from midpoint of start and end
    vec_start = geodetic_to_unit_vector(*start_coordinate)
    vec_end = geodetic_to_unit_vector(*end_coordinate)
    vec_mid = (vec_start + vec_end) / 2
    vec_mid /= np.linalg.norm(vec_mid)
    lon_0, lat_0 = unit_vector_to_geodetic(vec_mid)

    fig = plt.figure(figsize=(8, 8))
    m = Basemap(projection='ortho',
                resolution='l',
                lat_0=lat_0,
                lon_0=lon_0)

    # Default colors if not provided
    if path_colors is None:
        path_colors = plt.cm.tab10.colors

    # Draw the flight paths
    for i, path in enumerate(flight_paths):
        color = path_colors[i % len(path_colors)]
        xfp, yfp = m(path[:, 0], path[:, 1])
        m.plot(xfp, yfp, color=color, linewidth=path_width)

    # Draw start and end points in black
    xp, yp = m(start_coordinate[0], start_coordinate[1])
    xc, yc = m(end_coordinate[0], end_coordinate[1])
    m.scatter(xp, yp, marker='o', color='black', s=marker_size, zorder=5)
    m.scatter(xc, yc, marker='o', color='black', s=marker_size, zorder=5)

    # Path effects for text halo
    halo = [path_effects.withStroke(linewidth=3, foreground='white')]

    # Offset labels in points so they don't overlap the markers
    text_offset = (5, 5)  # x, y in points

    if start_city:
        plt.annotate(start_city, xy=(xp, yp), xytext=text_offset, textcoords='offset points',
                     color='black', fontsize=city_label_size, fontweight='bold',
                     path_effects=halo, zorder=6)
    if end_city:
        plt.annotate(end_city, xy=(xc, yc), xytext=text_offset, textcoords='offset points',
                     color='black', fontsize=city_label_size, fontweight='bold',
                     path_effects=halo, zorder=6)

    # Background
    m.bluemarble(scale=globe_scale)

    fig.tight_layout()
    save_path = f'{figure_path}earth_3d_multi_paths.pdf'
    fig.savefig(save_path, format='pdf', pad_inches=0.1, bbox_inches='tight')
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
    
#%%

def plot_jet_streams_comparison(lons,
                                lats,
                                u,
                                v,
                                speed,
                                flight_paths: dict,
                                start_end_labels: dict = None,  # e.g., {"start": "NYC", "end": "CPH"}
                                arrow_step=20,
                                title="Interpolated Jet Stream (250 hPa)",
                                figure_path: str = ''):

    # Ensure increasing latitude
    if lats[0] > lats[-1]:
        lats, u, v, speed = lats[::-1], u[::-1], v[::-1], speed[::-1]

    # --- determine region exactly from force field ---
    lon_min, lon_max = np.min(lons), np.max(lons)
    lat_min, lat_max = np.min(lats), np.max(lats)
    region_extent = [lon_min, lon_max, lat_min, lat_max]

    # Convert to numpy arrays
    lon_s = np.array(lons)
    lat_s = np.array(lats)
    u_s = np.array(u)
    v_s = np.array(v)
    speed_s = np.array(speed)

    lon_grid, lat_grid = np.meshgrid(lon_s, lat_s)

    # --- plotting ---
    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    # Base map
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.set_extent(region_extent, crs=ccrs.PlateCarree())
    
    # Plot contours
    cs = ax.contourf(lon_grid, lat_grid, speed_s, 60, cmap='coolwarm', transform=ccrs.PlateCarree())

    # Wind arrows
    # Normalize vectors to unit length
    norm = np.sqrt(u_s**2 + v_s**2)
    u_unit = u_s / (norm + 1e-6)
    v_unit = v_s / (norm + 1e-6)
    
    # Set desired arrow line length
    arrow_length = 0.02  # adjust this to make shafts shorter/longer
    
    ax.quiver(
        lon_grid[::arrow_step, ::arrow_step],
        lat_grid[::arrow_step, ::arrow_step],
        u_unit[::arrow_step, ::arrow_step] * arrow_length,
        v_unit[::arrow_step, ::arrow_step] * arrow_length,
        transform=ccrs.PlateCarree(),
        scale=1,         # keep scale=1 so length is controlled by arrow_length
        width=0.002,
        headlength=4,
        color='gray',
        alpha=0.7
    )


    # Flight paths
    colors = plt.cm.tab10.colors
    for i, (name, (travel_time, path)) in enumerate(flight_paths.items()):
        color = colors[i % len(colors)]

        # Convert travel time to h m
        hours = int(travel_time)
        minutes = int(round((travel_time - hours) * 60))
        legend_label = f"{name} ({hours}h {minutes}m)"

        # Smoothed line with label for legend
        ax.plot(path[:, 0], path[:, 1], linestyle='-', linewidth=2, color=color,
                transform=ccrs.PlateCarree(), label=legend_label)

        # --- start/end black dots ---
        ax.scatter(path[0, 0], path[0, 1], color='black', s=40, zorder=5, transform=ccrs.PlateCarree())
        ax.scatter(path[-1, 0], path[-1, 1], color='black', s=40, zorder=5, transform=ccrs.PlateCarree())

        # --- city labels with halo for better visibility ---
        offset_lon = 5.0  # push further out
        offset_lat = 5.0
        fontsize_city = 12
        halo_effect = [withStroke(linewidth=3, foreground='white')]
        
        if i == len(flight_paths.items())-1:
            if start_end_labels is not None:
                # Start point
                start_offset = offset_lon if path[0, 0] < (lon_min + lon_max)/2 else -offset_lon
                ax.text(path[0, 0] + start_offset, path[0, 1] + offset_lat,
                        start_end_labels.get("start", "Start"),
                        fontsize=fontsize_city, fontweight='bold', color='black', zorder=6,
                        transform=ccrs.PlateCarree(), va='center', path_effects=halo_effect)
    
                # End point
                end_offset = offset_lon if path[-1, 0] < (lon_min + lon_max)/2 else -offset_lon
                ax.text(path[-1, 0] + end_offset, path[-1, 1] + offset_lat,
                        start_end_labels.get("end", "End"),
                        fontsize=fontsize_city, fontweight='bold', color='black', zorder=6,
                        transform=ccrs.PlateCarree(), va='center', path_effects=halo_effect)

        # --- midpoint arrow showing direction ---
        mid_idx = len(path) // 2
        mid_point = path[mid_idx]
        if mid_idx > 0:
            dx = path[mid_idx + 1, 0] - path[mid_idx - 1, 0]
            dy = path[mid_idx + 1, 1] - path[mid_idx - 1, 1]
        else:
            dx = path[1, 0] - path[0, 0]
            dy = path[1, 1] - path[0, 1]

        ax.arrow(mid_point[0], mid_point[1], dx*0.05, dy*0.05,
                 head_width=2.0, head_length=1.5, fc="black", ec="black",
                 transform=ccrs.PlateCarree(), zorder=5)

    # --- legend (colors match curves automatically) ---
    ax.legend(loc='upper right', fontsize=14)

    plt.title(title, fontsize=14)
    
    # Plot contours
    # Shrink the main axes to leave space at the bottom
    pos = ax.get_position()  # Bbox(x0, y0, x1, y1)
    ax.set_position([pos.x0, pos.y0 + 0.05, pos.width, pos.height - 0.05])  # shrink height by 5% at bottom
    
    # Add full-width colorbar in the reserved bottom space
    cbar_ax = fig.add_axes([0.05, -0.02, 0.90, 0.03])  # left, bottom, width, height
    cbar = fig.colorbar(cs, cax=cbar_ax, orientation='horizontal', label='Wind Speed (km/h)')
    cbar.ax.tick_params(labelsize=14)                # Set font size of the tick labels
    
    fig.tight_layout()

    # Save figure
    save_path = f'{figure_path}earth_region_chart.pdf'
    fig.savefig(save_path, format='pdf', pad_inches=0.1, bbox_inches='tight', facecolor='white')
    plt.close()


