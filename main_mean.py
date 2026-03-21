#%% Modules

import os

import jax.numpy as jnp
from jax import jit, vmap

import geopandas as gpd
import airportsdata

from prettytable import PrettyTable

import argparse

from typing import List

from flight_model.download_jet_streams import (
    DownloadStaticJetStreams,
    DownloadTimeJetStreams,
    )

from flight_model.interpolation import (
    BilinearInterpolation,
    BicubicInterpolation,
    TrilinearInterpolation,
    TricubicInterpolation,
    )

from flight_model.conversion import (
    PositionEncoding,
    CoordinateConverter,
    )

from flight_model.manifold import (
    WGS84Earth_Chart,
    )

from flight_model.plotting import (
    plot_jet_streams_full_mean,
    plot_multiple_jet_streams_mean,
    plot_jet_streams_region_mean
    )

from flight_model.region_penalizer import (
    RegionPenalizer,
    GaussianRegionPenalizer,
    BoxRegionPenalizer,
    )

#%% Argeparse

#Tokyo: 35°41′22″N 139°41′30″E
#Copenhagen: 55°40′34″N 12°34′06″E
#New York: 40°42′46″N 74°00′22″W"
#Miami: 25°45′42″N 80°11′30″W

#Honolulu: 21°19′20″N 157°55′30″W
#Los Angeles: 33°56′33″N 118°24′29″W

def parse_args():
    parser = argparse.ArgumentParser()
    # File-paths
    parser.add_argument('--airports', default=[
        #"CPH","ARN","OSL","HEL","KEF","LHR","LGW","STN","MAN","DUB",
        #"AMS","BRU","CDG","ORY","FRA","MUC","BER","ZRH","GVA","VIE",
        #"PRG","BUD","WAW","KRK","ATH","SKG","FCO","MXP","LIN","BCN",
        #"MAD","PMI","LIS","OPO","IST","SAW","AYT","TLL","RIX","VNO",
        #"SOF","OTP","BEG","ZAG","LJU","SJJ","TGD","TIA","MLA","LCA",
        #"HND","NRT","KIX","ITM","NGO","ICN","GMP","PUS","PEK","PKX",
        #"PVG","SHA","CAN","SZX","CTU","HKG","MFM","TPE","TSA","KHH",
        #"SIN","KUL","BKK","DMK","HKT","SGN","HAN","DPS","CGK","SUB",
        #"MNL","CEB","DEL","BOM","BLR","HYD","MAA","CCU","KTM","DAC",
        #"CMB","DOH","DXB","DWC","AUH","SHJ","RUH","JED","AMM","TLV"
        "CPH","ARN","OSL","HEL","LHR","LGW","STN","MAN","DUB",
        "AMS","BRU","CDG","ORY","FRA","MUC","BER","ZRH","GVA","VIE",
        "PRG","BUD","WAW","KRK","ATH","SKG","FCO","MXP","LIN","BCN",
        "MAD","PMI","LIS","OPO","IST","SAW","AYT","TLL","RIX","VNO",
        "SOF","OTP","BEG","ZAG","LJU","SJJ","TGD","TIA","MLA","LCA",
        "PEK","PKX","PVG","SHA","CAN","SZX","CTU","HKG","MFM",
        "DEL","BOM","BLR","HYD","MAA","CCU","KTM","DAC",
        "CMB","DOH","DXB","DWC","AUH","SHJ","RUH","JED","AMM","TLV"
        ], type=list)
    parser.add_argument('--restricted_countries', default=["Mongolia"], #United Kingdom
                        type=list)
    parser.add_argument('--height', default=11.0,
                        type=float)
    parser.add_argument('--flight_speed', default=800., #900 km/h
                        type=float)
    parser.add_argument('--start_time', default='12:00',
                        type=str)
    parser.add_argument('--time_step', default=1.0,
                        type=float)
    parser.add_argument('--end_time', default=24.0,
                        type=str)
    parser.add_argument('--day', default='28',
                        type=str)
    parser.add_argument('--month', default='08',
                        type=str)
    parser.add_argument('--year', default='2025',
                        type=str)
    parser.add_argument('--pressure_level', default='250',
                        type=str)
    parser.add_argument('--interpolation_method', default="cubic", #linear, cubic
                        type=str)
    parser.add_argument('--region_points', default=100,
                        type=int)
    parser.add_argument('--grid_points', default=1000,
                        type=int)
    parser.add_argument('--max_iter', default=10, #10000
                        type=int)
    parser.add_argument('--max_step_size', default=0.1, #10000
                        type=float)
    parser.add_argument('--lam', default=20.0, #20.0, #10000
                        type=float)
    parser.add_argument('--tolerance', default=0.0,
                        type=float)
    parser.add_argument('--figure_path', default='../output_figures/',
                        type=str)
    parser.add_argument('--data_path', default='../data_output/',
                        type=str)

    args = parser.parse_args()
    return args

#%% Linear Interpolation benchmark

def linear_geodetic_shortest(start, end, T):
    """
    Linear interpolation in geodetic coordinates, shortest path for longitude.
    
    Args:
        start, end: (lon, lat) in degrees
        T: number of intervals → T+1 points
    Returns:
        Array (T+1, 2)
    """
    start = jnp.array(start)
    end = jnp.array(end)
    
    # Latitude: normal linear interpolation
    lat_start, lat_end = start[1], end[1]
    
    # Longitude: compute shortest delta
    lon_start, lon_end = start[0], end[0]
    delta_lon = lon_end - lon_start
    # Wrap into [-180, 180]
    delta_lon = ((delta_lon + 180) % 360) - 180
    
    # Linear interpolation weights
    t = jnp.linspace(0.0, 1.0, T+1).reshape(-1,1)
    
    lon_interp = lon_start + t[:,0] * delta_lon
    lat_interp = lat_start + t[:,0] * (lat_end - lat_start)
    
    return jnp.stack([lon_interp, lat_interp], axis=1)

def compute_travel_time_linear_curve(curve_geodetic, speed_kmh, earth_model):
    """
    Compute travel time along a linear geodetic curve with constant speed.
    
    Args:
        curve_geodetic: array of shape (T+1, 2) in (lon, lat) degrees
        speed_kmh: travel speed in km/h
        earth_model: instance of WGS84Earth_Euclidean
    
    Returns:
        travel_time in hours
    """
    # Convert all geodetic points to ECEF
    curve_ecef = jnp.stack([earth_model.geodetic_to_ecef(pt) for pt in curve_geodetic])
    
    # Compute distances between consecutive points
    deltas = curve_ecef[1:] - curve_ecef[:-1]   # shape (T, 3)
    segment_lengths = jnp.linalg.norm(deltas, axis=1)  # km
    
    # Total distance
    total_distance = jnp.sum(segment_lengths)
    
    # Travel time = distance / speed
    travel_time = total_distance / speed_kmh   # hours
    
    return travel_time


#%% Main code

def run_time_dependent_jet_stream(args, data_coordinates, penalizer, full_coords):
    
    figure_path = ''.join((args.figure_path, 'mean/time_dependent_jet_stream/'))
    data_path = ''.join((args.data_path, 'time_dependent_jet_stream/'))
    
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)
        
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    
    converter = PositionEncoding()
    
    data_coordinates = jnp.array([converter(coord) for coord in data_coordinates])
    
    data_loader = DownloadTimeJetStreams(start_time=args.start_time,
                                         time_step = args.time_step,     # in hours
                                         duration_hours = args.end_time,     # in hours
                                         day=args.day,
                                         month=args.month,
                                         year=args.year,
                                         pressure_level=args.pressure_level,
                                         area = (90, -180, -90, 180),
                                         save_path=data_path,
                                         )
    
    times, lons, lats, u, v = data_loader()

    # Convert everything to jax arrays
    lons, lats, u, v = jnp.array(lons), jnp.array(lats), jnp.array(u), jnp.array(v)
    times = jnp.array(times)  # hours after first
    speed = jnp.sqrt(u**2 + v**2)
    
    if args.interpolation_method == "linear":
        # Trilinear interpolation along time, lat, lon
        interpolation_speed = TrilinearInterpolation(lats, lons, times, speed)
        interpolation_u     = TrilinearInterpolation(lats, lons, times, u)
        interpolation_v     = TrilinearInterpolation(lats, lons, times, v)
    elif args.interpolation_method == "cubic":
        # For tricubic, you would use TricubicInterpolation (if implemented)
        interpolation_speed = TricubicInterpolation(lons, lats, times, speed)
        interpolation_u     = TricubicInterpolation(lons, lats, times, u)
        interpolation_v     = TricubicInterpolation(lons, lats, times, v)
    else:
        raise ValueError(f"Only linear and cubic interpolation methods supported. Invalid choice: {args.interpolation_method}")
    
    # Interpolate on a fine grid in (lon, lat, time)
    lon_fine, lat_fine, time_fine, speed_fine = interpolation_speed.interpolate_field(grid_size=(800, 400, 50))
    _, _, _, u_fine = interpolation_u.interpolate_field(grid_size=(800, 400, 50))
    _, _, _, v_fine = interpolation_v.interpolate_field(grid_size=(800, 400, 50))

    field_fun = lambda t, x: jnp.array([
        interpolation_u(x[0], x[1], t),  # u component at (time, lat, lon)
        interpolation_v(x[0], x[1], t)   # v component at (time, lat, lon)
    ])
    
    
    manifold_jet_riemannian_stream = WGS84Earth_Chart(force_field=None, 
                                                      reg_fun=penalizer,
                                                      flight_speed=args.flight_speed, 
                                                      lam = 0.0,
                                                      height=args.height,
                                                      grid_points=args.grid_points,
                                                      max_iter=args.max_iter,
                                                      max_step_size=1.0,
                                                      tol=args.tolerance,
                                                      mean_type="start",
                                                      )
    
    manifold_jet_full_stream = WGS84Earth_Chart(force_field=field_fun, 
                                                reg_fun=penalizer,
                                                flight_speed=args.flight_speed, 
                                                lam = 0.0,
                                                height=args.height,
                                                grid_points=args.grid_points,
                                                max_iter=args.max_iter,
                                                max_step_size=1.0,
                                                tol=args.tolerance,
                                                mean_type="start",
                                                )
    
    manifold_region_start_jet_stream = WGS84Earth_Chart(force_field=field_fun, 
                                                  reg_fun=penalizer,
                                                  flight_speed=args.flight_speed, 
                                                  lam = args.lam,
                                                  height=args.height,
                                                  grid_points=args.grid_points,
                                                  max_iter=args.max_iter,
                                                  max_step_size=args.max_step_size,
                                                  tol=args.tolerance,
                                                  mean_type="start",
                                                  )
    manifold_region_end_jet_stream = WGS84Earth_Chart(force_field=field_fun, 
                                                      reg_fun=penalizer,
                                                      flight_speed=args.flight_speed, 
                                                      lam = args.lam,
                                                      height=args.height,
                                                      grid_points=args.grid_points,
                                                      max_iter=args.max_iter,
                                                      max_step_size=args.max_step_size,
                                                      tol=args.tolerance,
                                                      mean_type="end",
                                                      )
    
    travel_riemannian_times, flight_riemannian_paths = manifold_jet_riemannian_stream.mean(t0=.0,
                                                                                           z_obs=data_coordinates
                                                                                           )
    travel_riemannian_times = vmap(manifold_jet_full_stream.travel_time)(flight_riemannian_paths)
    
    travel_full_times, flight_full_paths = manifold_jet_full_stream.mean(t0=.0,
                                                                         z_obs=data_coordinates
                                                                         )
    travel_full_times = vmap(manifold_jet_full_stream.travel_time)(flight_full_paths)
    
    travel_start_pen_times, flight_start_pen_paths = manifold_region_start_jet_stream.mean(t0=.0,
                                                                                           z_obs=data_coordinates
                                                                                           )
    travel_start_pen_times = vmap(manifold_jet_full_stream.travel_time)(flight_start_pen_paths)
    
    travel_end_pen_times, flight_end_pen_paths = manifold_region_end_jet_stream.mean(t0=.0,
                                                                                     z_obs=data_coordinates
                                                                                     )
    travel_end_pen_times = vmap(manifold_jet_full_stream.travel_time)(flight_end_pen_paths)
    
    plot_riemannian_paths = {f'curve_{idx}': [time[-1], curve, "red"] \
                             for idx, (time,curve) in enumerate(zip(travel_riemannian_times, 
                                                                    flight_riemannian_paths), 
                                                                start=0)
                             }
    plot_full_paths = {f'curve_{idx}': [time[-1], curve, "red"] \
                       for idx, (time,curve) in enumerate(zip(travel_full_times, 
                                                              flight_full_paths), 
                                                          start=0)
                       }
    plot_start_pen_paths = {f'curve_{idx}': [time[-1], curve, "red"] \
                             for idx, (time,curve) in enumerate(zip(travel_start_pen_times, 
                                                                flight_start_pen_paths), 
                                                                start=0)
                             }
    plot_end_pen_paths = {f'curve_{idx}': [time[-1], curve, "red"] \
                          for idx, (time,curve) in enumerate(zip(travel_end_pen_times, 
                                                                 flight_end_pen_paths), 
                                                             start=0)
                          }

    total_riemannian_time = jnp.mean(jnp.abs(travel_riemannian_times[:,-1]))
    hours = int(total_riemannian_time)
    minutes = int(round((total_riemannian_time - hours) * 60))
    legend_riemannian_label = f"Riemannian mean ({hours}h {minutes}m)"
    
    total_full_time = jnp.mean(jnp.abs(travel_full_times[:,-1]))
    hours = int(total_full_time)
    minutes = int(round((total_full_time - hours) * 60))
    legend_full_label = f"Time-dependent Finslerian start mean ({hours}h {minutes}m)"
    
    total_start_pen_time = jnp.mean(jnp.abs(travel_start_pen_times[:,-1]))
    hours = int(total_start_pen_time)
    minutes = int(round((total_start_pen_time - hours) * 60))
    legend_start_pen_label = f"Pen. Time-dependent Finslerian start mean ({hours}h {minutes}m)"
    
    total_end_pen_time = jnp.mean(jnp.abs(travel_end_pen_times[:,-1]))
    hours = int(total_end_pen_time)
    minutes = int(round((total_end_pen_time - hours) * 60))
    legend_end_pen_label = f"Pen. Time-dependent Finslerian end mean ({hours}h {minutes}m)"
    
    plot_jet_streams_full_mean(lon_fine, lat_fine, u_fine[0], v_fine[0], speed_fine[0], 
                               flight_paths = plot_riemannian_paths,
                               full_coords=full_coords,
                               average_flight_time=legend_riemannian_label,
                               figure_path=f"{figure_path}riemannian_mean",
                               title=None,
                               )
    plot_jet_streams_full_mean(lon_fine, lat_fine, u_fine[0], v_fine[0], speed_fine[0], 
                               flight_paths = plot_full_paths,
                               full_coords=full_coords,
                               average_flight_time=legend_full_label,
                               figure_path=f"{figure_path}start_mean",
                               title=None,
                               )
    plot_jet_streams_full_mean(lon_fine, lat_fine, u_fine[0], v_fine[0], speed_fine[0], 
                               flight_paths = plot_start_pen_paths,
                               full_coords=full_coords,
                               average_flight_time=legend_start_pen_label,
                               figure_path=f"{figure_path}pen_start_mean",
                               title=None,
                               )
    plot_jet_streams_full_mean(lon_fine, lat_fine, u_fine[0], v_fine[0], speed_fine[0], 
                               flight_paths = plot_end_pen_paths,
                               full_coords=full_coords,
                               average_flight_time=legend_end_pen_label,
                               figure_path=f"{figure_path}pen_end_mean",
                               title=None,
                               )
    
    plot_jet_streams_region_mean(lon_fine, lat_fine, u_fine[0], v_fine[0], speed_fine[0], 
                               flight_paths = plot_riemannian_paths,
                               full_coords=full_coords,
                               average_flight_time=legend_riemannian_label,
                               figure_path=f"{figure_path}riemannian_mean_region",
                               title=None,
                               )
    plot_jet_streams_region_mean(lon_fine, lat_fine, u_fine[0], v_fine[0], speed_fine[0], 
                               flight_paths = plot_full_paths,
                               full_coords=full_coords,
                               average_flight_time=legend_full_label,
                               figure_path=f"{figure_path}start_mean_region",
                               title=None,
                               )
    plot_jet_streams_region_mean(lon_fine, lat_fine, u_fine[0], v_fine[0], speed_fine[0], 
                               flight_paths = plot_start_pen_paths,
                               full_coords=full_coords,
                               average_flight_time=legend_start_pen_label,
                               figure_path=f"{figure_path}pen_start_mean_region",
                               title=None,
                               )
    plot_jet_streams_region_mean(lon_fine, lat_fine, u_fine[0], v_fine[0], speed_fine[0], 
                               flight_paths = plot_end_pen_paths,
                               full_coords=full_coords,
                               average_flight_time=legend_end_pen_label,
                               figure_path=f"{figure_path}pen_end_mean_region",
                               title=None,
                               )
    
    
    plot_multiple_jet_streams_mean(lon_fine, lat_fine, u_fine[0], v_fine[0], speed_fine[0], 
                               list_of_flight_paths = [plot_riemannian_paths, 
                                                       plot_full_paths, 
                                                       plot_start_pen_paths,
                                                       plot_end_pen_paths],
                               full_coords=full_coords,
                               average_flight_times=[legend_riemannian_label,
                                                     legend_full_label,
                                                     legend_start_pen_label, 
                                                     legend_end_pen_label,
                                                     ],
                               figure_path=f"{figure_path}flight_means",
                               )
    
    return

#%% Region avoidance

def construct_region_avoidance(restricted_countries:List,
                               tau = 0.01,
                                delta = 0.1,
                                scale = 1000.0,
                               ):
    
    if restricted_countries:
        # Load GeoJSON of countries
        url = "https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson"
        world = gpd.read_file(url)
        
        penalizer_funs = []
        full_coords = []
        for country_name in restricted_countries:
            country = world[world['name'] == country_name]
            
            # Extract the boundary
            boundary = country.geometry.boundary
        
            coords = []
        
            for geom in boundary:
                if geom.geom_type == 'LineString':
                    coords.extend(list(geom.coords))
        
                elif geom.geom_type == 'MultiLineString':
                    for line in geom.geoms:   # <-- FIX HERE
                        coords.extend(list(line.coords))
                        
            
            
            # coords: jnp.array (lon, lat) as loaded from GeoJSON
            coords = jnp.array(coords)  # shape (N, 2)
            sub_points = len(coords) // args.region_points
            coords = coords[::sub_points]

            # Step 1: Ensure Northern Hemisphere
            coords = coords.at[:, 1].set(jnp.abs(coords[:, 1]))  # latitude
            
            # Step 2: Ensure longitude in -180..180
            coords = coords.at[:, 0].set(((coords[:, 0] + 180) % 360) - 180)

            full_coords.append(coords)
            #reg_fun = GaussianRegionPenalizer(vertices=jnp.array(coords),
            #                                  scale = 10.0,
            #                                  regularization = 1e-3,
            #                                  margin=0.1,
            #                                  )
            reg_fun = RegionPenalizer(vertices=jnp.array(coords),#+180.,
                                      tau=tau,
                                      delta=delta   # 10–12 km clearance
                                      )

            #reg_fun = BoxRegionPenalizer(vertices=jnp.array(coords),#+180.,
            #                             delta=delta   # 10–12 km clearance
            #                             )
            penalizer_funs.append(jit(reg_fun))
        return lambda x: sum(f(x) for f in penalizer_funs), full_coords
    else:
        return None, None

#%% Run main

if __name__ == "__main__":
    
    args = parse_args()
    
    converter = CoordinateConverter()

    airports = airportsdata.load("IATA")
    
    
    city_names = [airports[air]['city'] for air in args.airports]
    lats = [airports[air]['lat'] for air in args.airports]
    lons = [airports[air]['lon'] for air in args.airports]
    airports_data = [airports[air] for air in args.airports]
    data_coordinates = [converter(lat, lon) for lat,lon in zip(lats, lons)]
        
    penalizer, full_coords = construct_region_avoidance(restricted_countries=args.restricted_countries,
                                                        tau = 0.5, #0.05,
                                                        delta = 0.1,
                                                        scale = 20.0,
                                                        )

    run_time_dependent_jet_stream(args, data_coordinates, penalizer, full_coords)
    

    
        