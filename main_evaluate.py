#%% Modules

import os

import jax.numpy as jnp
from jax import jit

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
    plot_earth_chart,
    plot_3d_earth_paths,
    plot_earth_region_chart,
    plot_jet_streams_region,
    plot_jet_streams_full,
    plot_time_paths_region,
    animate_flight_dynamic,
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
    parser.add_argument('--start_coordinate', default="KMIA", #JFK
                        type=str)
    parser.add_argument('--end_coordinate', default="LFPG", #CPH
                        type=str)
    parser.add_argument('--start_city', default="Miami",
                        type=str)
    parser.add_argument('--end_city', default="Paris",
                        type=str)
    parser.add_argument('--restricted_countries', default=[], #United Kingdom
                        type=list)
    parser.add_argument('--height', default=11.0,
                        type=float)
    parser.add_argument('--flight_speed', default=800., #900 km/h
                        type=float)
    parser.add_argument('--start_time', default='17:00',
                        type=str)
    parser.add_argument('--time_step', default=1.0,
                        type=float)
    parser.add_argument('--end_time', default=24.0,
                        type=str)
    parser.add_argument('--day', default='01',
                        type=str)
    parser.add_argument('--month', default='02',
                        type=str)
    parser.add_argument('--year', default='2026',
                        type=str)
    parser.add_argument('--pressure_level', default='250',
                        type=str)
    parser.add_argument('--jet_stream_method', default="time_dependent", #static, none, time_dependent
                        type=str)
    parser.add_argument('--interpolation_method', default="cubic", #linear, cubic
                        type=str)
    parser.add_argument('--region_points', default=100,
                        type=int)
    parser.add_argument('--num_waypoints', default=5,
                        type=int)
    parser.add_argument('--grid_points', default=1000,
                        type=int)
    parser.add_argument('--max_iter', default=1000, #10000
                        type=int)
    parser.add_argument('--max_step_size', default=0.01, #10000
                        type=float)
    parser.add_argument('--lam', default=20.0, #10000
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

def run_static_jet_stream(args, penalizer, full_coords):
    
    figure_path = ''.join((args.figure_path, 'static_jet_stream/'))
    data_path = ''.join((args.data_path, 'static__jet_stream/'))
    
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)
        
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    
    converter = PositionEncoding()
    
    start_coordinate = jnp.array(converter(args.start_coordinate))
    end_coordinate = jnp.array(converter(args.end_coordinate))
    
    data_loader = DownloadStaticJetStreams(time=args.start_time,
                                           day=args.day,
                                           month=args.month,
                                           year=args.year,
                                           pressure_level=args.pressure_level,
                                           area = (90, -180, -90, 180),
                                           save_path=data_path,
                                           )
    
    lons, lats, u, v = data_loader()
    
    lons, lats, u, v = jnp.array(lons), jnp.array(lats), jnp.array(u), jnp.array(v)
    speed = jnp.sqrt(u**2+v**2)
    
    if args.interpolation_method == "linear":
        interpolation_speed = BilinearInterpolation(lons, lats, speed)
        interpolation_u = BilinearInterpolation(lons, lats, u)
        interpolation_v = BilinearInterpolation(lons, lats, v)
    elif args.interpolation_method == "cubic":
        interpolation_speed = BicubicInterpolation(lons, lats, speed)
        interpolation_u = BicubicInterpolation(lons, lats, u)
        interpolation_v = BicubicInterpolation(lons, lats, v)
    else:
        raise ValueError(f"Only linear and cubic interpolation methods supported. Invalid choice: {args.interpolation_method}")
    
    lon_fine, lat_fine, speed_fine = interpolation_speed.interpolate_field(grid_size=(800,400))
    _, _, u_fine = interpolation_u.interpolate_field(grid_size=(800,400))
    _, _, v_fine = interpolation_v.interpolate_field(grid_size=(800,400))

    field_fun = lambda t,x: jnp.array([interpolation_u(x[0], x[1]),
                                       interpolation_v(x[0], x[1])])
    
    manifold_jet_stream = WGS84Earth_Chart(force_field=field_fun, 
                                           reg_fun=penalizer,
                                           flight_speed=args.flight_speed, 
                                           height=args.height,
                                           lam = 0.0,
                                           grid_points=args.grid_points,
                                           max_iter=args.max_iter,
                                           max_step_size=1.0,
                                           tol=args.tolerance,
                                           )
    
    travel_time, flight_path = manifold_jet_stream.geodesic(start_coordinate, end_coordinate)
    travel_time_reverse, flight_path_reverse = manifold_jet_stream.geodesic(end_coordinate, start_coordinate)
    
    manifold_no_jet_stream = WGS84Earth_Chart(force_field=None, 
                                              reg_fun=penalizer,
                                              flight_speed=args.flight_speed, 
                                              height=args.height,
                                              lam = 0.0,
                                              grid_points=args.grid_points,
                                              max_iter=args.max_iter,
                                              max_step_size=1.0,
                                              tol=args.tolerance,
                                              )
    
    travel_time_no_js, flight_path_no_js = manifold_no_jet_stream.geodesic(start_coordinate, end_coordinate)
    travel_time_no_js = manifold_jet_stream.travel_time(flight_path_no_js)
    
    
    
    flight_path_euclidean = linear_geodetic_shortest(start_coordinate, end_coordinate, args.grid_points)
    travel_time_euclidean = manifold_jet_stream.travel_time(flight_path_euclidean)
    #travel_time_euclidean  = compute_travel_time_linear_curve(flight_path_euclidean, 
    #                                                          args.flight_speed, 
    #                                                          manifold_no_jet_stream,
    #                                                          )
    
    
    plot_earth_chart(start_coordinate, end_coordinate, flight_path,
                     figure_path=args.figure_path)
    plot_3d_earth_paths(start_coordinate, end_coordinate, [flight_path, flight_path_no_js, flight_path_euclidean],
                        colors=["red","orange","green"],
                        start_city=args.start_city, end_city=args.end_city,
                        figure_path=figure_path)
    
    plot_jet_streams_region(lon_fine, lat_fine, u_fine, v_fine, speed_fine, 
                     flight_paths = {'With Jet Stream': [travel_time[-1], flight_path, "red"],
                                    'No Jet Stream': [travel_time_no_js[-1], flight_path_no_js, "orange"],
                                    'Straight line': [travel_time_euclidean[-1], flight_path_euclidean, "green"]},
                     start_end_labels = {'start': args.start_city,
                                         'end': args.end_city,
                                         },
                     full_coords=full_coords,
                     figure_path=figure_path,
                     title="Optimal Flight Path (250 hPa)"
                     )
    
    plot_jet_streams_full(lon_fine, lat_fine, u_fine, v_fine, speed_fine, 
                     flight_paths = {'With Jet Stream': [travel_time[-1], flight_path, "red"],
                                    'No Jet Stream': [travel_time_no_js[-1], flight_path_no_js, "orange"],
                                    'Straight line': [travel_time_euclidean[-1], flight_path_euclidean, "green"]},
                     start_end_labels = {'start': args.start_city,
                                         'end': args.end_city,
                                         },
                     full_coords=full_coords,
                     figure_path=figure_path,
                     title="Optimal Flight Path (250 hPa)"
                     )
    
    plot_jet_streams_region(lon_fine, lat_fine, u_fine, v_fine, speed_fine, 
                     flight_paths = {f'{args.end_city}-{args.start_city}': [travel_time_reverse[-1], flight_path_reverse, "purple"],
                                     f'{args.start_city}-{args.end_city}': [travel_time[-1], flight_path, "red"],
                                     },
                     start_end_labels = {'start': args.start_city,
                                         'end': args.end_city,
                                         },
                     full_coords=full_coords,
                     figure_path=f'{figure_path}reverse_path',
                     title="Optimal Flight Path (250 hPa)"
                     )
    
    plot_3d_earth_paths(start_coordinate, end_coordinate, [flight_path_reverse, flight_path],
                        colors=["purple","red"],
                        start_city=args.start_city, end_city=args.end_city,
                        figure_path=f'{figure_path}reverse_path',
                        )
    
    return travel_time, flight_path

#%% Main code

def run_time_dependent_jet_stream(args, penalizer, full_coords):
    
    figure_path = ''.join((args.figure_path, 'evaluate/time_dependent_jet_stream/'))
    data_path = ''.join((args.data_path, 'evaluate/time_dependent_jet_stream/'))
    
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)
        
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    
    converter = PositionEncoding()
    
    start_coordinate = jnp.array(converter(args.start_coordinate))
    end_coordinate = jnp.array(converter(args.end_coordinate))
    
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
    
    manifold_jet_stream = WGS84Earth_Chart(force_field=field_fun, 
                                           reg_fun=penalizer,
                                           flight_speed=args.flight_speed, 
                                           lam = 0.0,
                                           height=args.height,
                                           grid_points=args.grid_points,
                                           max_iter=args.max_iter,
                                           max_step_size=1.0,
                                           tol=args.tolerance,
                                           )
    manifold_region_jet_stream = WGS84Earth_Chart(force_field=field_fun, 
                                                  reg_fun=penalizer,
                                                  flight_speed=args.flight_speed, 
                                                  lam = args.lam,
                                                  height=args.height,
                                                  grid_points=args.grid_points,
                                                  max_iter=args.max_iter,
                                                  max_step_size=args.max_step_size,
                                                  tol=args.tolerance,
                                                  )
    
    travel_time, flight_path = manifold_jet_stream.geodesic(start_coordinate, 
                                                            end_coordinate)
    travel_region_time, flight_region_path = manifold_region_jet_stream.geodesic(start_coordinate, 
                                                                                 end_coordinate)
    
    travel_time_reverse, flight_path_reverse = manifold_jet_stream.geodesic(end_coordinate, start_coordinate)
    
    manifold_no_jet_stream = WGS84Earth_Chart(force_field=None, 
                                              reg_fun=penalizer,
                                              flight_speed=args.flight_speed, 
                                              height=args.height,
                                              lam = 0.0,
                                              grid_points=args.grid_points,
                                              max_iter=args.max_iter,
                                              max_step_size=1.0,
                                              tol=args.tolerance,
                                              )
    
    travel_time_no_js, flight_path_no_js = manifold_no_jet_stream.geodesic(start_coordinate, end_coordinate)
    travel_time_no_js = manifold_jet_stream.travel_time(flight_path_no_js)
    
    flight_path_euclidean = linear_geodetic_shortest(start_coordinate, end_coordinate, args.grid_points)
    travel_time_euclidean = manifold_jet_stream.travel_time(flight_path_euclidean)
    
    animate_flight_dynamic(
        lon_fine,
        lat_fine,
        travel_time,
        interpolation_speed,
        interpolation_u,
        interpolation_v,
        flight_path[:,0],
        flight_path[:,1],
        figure_path=figure_path,
        arrow_step=20,
        fps=10
          )
    
    plot_earth_chart(start_coordinate, end_coordinate, flight_path,
                     figure_path=figure_path)
    plot_3d_earth_paths(start_coordinate, end_coordinate, [flight_path, flight_path_no_js, flight_path_euclidean],
                        colors=["red","orange","green"],
                        start_city=args.start_city, end_city=args.end_city,
                        figure_path=figure_path,
                        )

    plot_jet_streams_region(lon_fine, lat_fine, u_fine[0], v_fine[0], speed_fine[0], 
                     flight_paths = {'With Jet Stream': [travel_time[-1], flight_path, "red"],
                                    'No Jet Stream': [travel_time_no_js[-1], flight_path_no_js, "orange"],
                                    'Straight line': [travel_time_euclidean[-1], flight_path_euclidean, "green"]},
                     start_end_labels = {'start': args.start_city,
                                         'end': args.end_city,
                                         },
                     full_coords=None,
                     figure_path=figure_path,
                     title="Optimal Flight Path (250 hPa)"
                     )
    
    plot_jet_streams_region(lon_fine, lat_fine, u_fine[0], v_fine[0], speed_fine[0], 
                     flight_paths = {'Finslerian': [travel_time[-1], flight_path, "red"],
                                    'Riemannian': [travel_time_no_js[-1], flight_path_no_js, "orange"],
                                    'Straight line': [travel_time_euclidean[-1], flight_path_euclidean, "green"]},
                     start_end_labels = {'start': args.start_city,
                                         'end': args.end_city,
                                         },
                     full_coords=None,
                     figure_path=figure_path,
                     title="Optimal Flight Path (250 hPa)"
                     )
    
    plot_jet_streams_full(lon_fine, lat_fine, u_fine[0], v_fine[0], speed_fine[0], 
                     flight_paths = {'Finslerian': [travel_time[-1], flight_path, "red"],
                                    'Region avoidance': [travel_region_time[-1], flight_region_path, "orange"],
                                    },
                     start_end_labels = {'start': args.start_city,
                                         'end': args.end_city,
                                         },
                     full_coords=full_coords,
                     figure_path=f"{figure_path}region_constraint",
                     title="Optimal Flight Path (250 hPa)"
                     )
    
    plot_jet_streams_region(lon_fine, lat_fine, u_fine[0], v_fine[0], speed_fine[0], 
                     flight_paths = {'Finslerian': [travel_time[-1], flight_path, "red"],
                                    'Region avoidance': [travel_region_time[-1], flight_region_path, "green"],
                                    },
                     start_end_labels = {'start': args.start_city,
                                         'end': args.end_city,
                                         },
                     full_coords=full_coords,
                     figure_path=f"{figure_path}region_constraint",
                     title="Optimal Flight Path (250 hPa)"
                     )
    
    plot_jet_streams_region(lon_fine, lat_fine, u_fine[0], v_fine[0], speed_fine[0], 
                     flight_paths = {f'{args.end_city}-{args.start_city}': [travel_time_reverse[-1], flight_path_reverse, "purple"],
                                     f'{args.start_city}-{args.end_city}': [travel_time[-1], flight_path, "red"],
                                     },
                     start_end_labels = {'start': args.start_city,
                                         'end': args.end_city,
                                         },
                     full_coords=None,
                     figure_path=f'{figure_path}reverse_path',
                     title="Optimal Flight Path (250 hPa)"
                     )
    
    plot_3d_earth_paths(start_coordinate, end_coordinate, [flight_path_reverse, flight_path],
                        colors=["purple","red"],
                        start_city=args.start_city, end_city=args.end_city,
                        figure_path=f'{figure_path}reverse_path',
                        )
    
    plot_time_paths_region(lon_fine, lat_fine, u_fine[0], v_fine[0], speed_fine[0], 
                     flight_paths = {'curve1': ["Time-dependent Finslerian", travel_time[int(len(travel_time)*0.25)], 
                                                    flight_path[:int(len(travel_time)*0.25)], "red"],
                                     'curve2': ["Time-dependent Finslerian", travel_time[int(len(travel_time)*0.50)], 
                                                                    flight_path[:int(len(travel_time)*0.50)], "red"],
                                     'curve3': ["Time-dependent Finslerian", travel_time[int(len(travel_time)*0.75)], 
                                                                    flight_path[:int(len(travel_time)*0.75)], "red"],
                                     'curve4': ["Time-dependent Finslerian", travel_time[-1], 
                                                                    flight_path, "red"],
                                    },
                     start_end_labels = {'start': args.start_city,
                                         'end': args.end_city,
                                         },
                     start_city = flight_path[0],
                     end_city = flight_path[-1],
                     full_coords=None,
                     figure_path=figure_path,
                     title=""
                     )
    
    return travel_time, flight_path

#%% Main code

def run_no_jet_field(args, penalizer, full_coords):
    
    figure_path = ''.join((args.figure_path, 'no_jet_stream/'))
    
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)
        
    converter = PositionEncoding()
    
    start_coordinate = jnp.array(converter(args.start_coordinate))
    end_coordinate = jnp.array(converter(args.end_coordinate))
    
    manifold = WGS84Earth_Chart(force_field=None, 
                                reg_fun=penalizer,
                                flight_speed=args.flight_speed, 
                                height=args.height,
                                grid_points=args.grid_points,
                                max_iter=args.max_iter,
                                max_step_size=args.max_step_size,
                                tol=args.tolerance,
                                )
    
    travel_time, flight_path = manifold.geodesic(start_coordinate, end_coordinate)
    
    plot_earth_chart(start_coordinate, end_coordinate, flight_path,
                     figure_path=figure_path)
    plot_earth_region_chart(start_coordinate, end_coordinate, flight_path,
                            figure_path=figure_path)
    
    
    plot_3d_earth_paths(start_coordinate, end_coordinate, [flight_path],
                        colors=["red"],
                        start_city=args.start_city, end_city=args.end_city,
                        figure_path=figure_path,
                        )
    
    return travel_time, flight_path

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

    airports = airportsdata.load("ICAO")
    
    try:
        start_airport = airports[args.start_coordinate]
        
        lat = start_airport['lat']
        lon = start_airport['lon']
        
        args.start_coordinate = converter(lat, lon)
        
        start_city = start_airport['city']
        
        
        # Create table
        table = PrettyTable()
        table.field_names = ["Info", "Departure Airport"]
        
        for key, value in start_airport.items():
            table.add_row([key, value])
            
        print(table)
        
        
    except:
        start_coordinate = args.start_coordinate
        
    try:
        end_airport = airports[args.end_coordinate]
        
        lat = end_airport['lat']
        lon = end_airport['lon']
        
        args.end_coordinate = converter(lat, lon)
        
        end_city = end_airport['city']
        
        # Create table
        table = PrettyTable()
        table.field_names = ["Info", "Arrival Airport"]
        
        for key, value in end_airport.items():
            table.add_row([key, value])
            
        print(table)
        
    except:
        end_coordinate = args.end_coordinate
        
    penalizer, full_coords = construct_region_avoidance(restricted_countries=args.restricted_countries,
                                                        tau = 0.5, #0.05,
                                                        delta = 0.1,
                                                        scale = 20.0,
                                                        )

    if args.jet_stream_method == "none":
        travel_time, flight_path = run_no_jet_field(args, penalizer, full_coords)
    elif args.jet_stream_method == "static":
        travel_time, flight_path = run_static_jet_stream(args, penalizer, full_coords)
    elif args.jet_stream_method == "time_dependent":
        travel_time, flight_path = run_time_dependent_jet_stream(args, penalizer, full_coords)
    else:
        raise ValueError(f"Invalid jet_stream_method {args.jet_stream_method}. Possible methods are: none, static, time_dependent")
        
    if travel_time is not None:
        table = PrettyTable()
        table.field_names = ["Estimated Time", "Waypoint"]
        for ts, curve in zip(travel_time[::(args.grid_points//args.num_waypoints)], 
                             flight_path[::(args.grid_points//args.num_waypoints)]):
            table.add_row([f"{ts:.2f}", converter(curve[1], curve[0])])
        
        print(table)
        
    

    
        