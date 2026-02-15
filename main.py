#%% Modules

import os

import jax.numpy as jnp

import argparse

from flight_model.download_jet_streams import (
    DownloadJetStreams
    )

from flight_model.interpolation import (
    BilinearInterpolation
    )

from flight_model.conversion import (
    PositionEncoding,
    )

from flight_model.manifold import (
    WGS84Earth_Chart,
    )

from flight_model.plotting import (
    plot_earth_chart,
    plot_3d_earth,
    plot_3d_earth_multi_paths,
    plot_earth_region_chart,
    plot_jet_streams,
    plot_jet_streams_comparison,
    )

#%% Argeparse

#Tokyo: 35°41′22″N 139°41′30″E
#Copenhagen: 55°40′34″N 12°34′06″E
#New York: 40°42′46″N 74°00′22″W"
#Miami: 25°45′42″N 80°11′30″W

def parse_args():
    parser = argparse.ArgumentParser()
    # File-paths
    parser.add_argument('--start_coordinate', default="40°42′46″N 74°00′22″W",
                        type=str)
    parser.add_argument('--end_coordinate', default="55°40′34″N 12°34′06″E",
                        type=str)
    parser.add_argument('--start_city', default="New York",
                        type=str)
    parser.add_argument('--end_city', default="Copenhagen",
                        type=str)
    parser.add_argument('--height', default=11.0,
                        type=float)
    parser.add_argument('--flight_speed', default=900., #900 km/h
                        type=float)
    parser.add_argument('--time', default='12:00',
                        type=str)
    parser.add_argument('--day', default='28',
                        type=str)
    parser.add_argument('--month', default='08',
                        type=str)
    parser.add_argument('--year', default='2025',
                        type=str)
    parser.add_argument('--pressure_level', default='250',
                        type=str)
    parser.add_argument('--use_jet_stream', default=True,
                        type=bool)
    parser.add_argument('--grid_points', default=1000,
                        type=int)
    parser.add_argument('--max_iter', default=10000,
                        type=int)
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

def run_wtih_jet_stream(args):
    
    if not os.path.exists(args.figure_path):
        os.makedirs(args.figure_path)
        
    if not os.path.exists(args.data_path):
        os.makedirs(args.data_path)
    
    converter = PositionEncoding()
    
    start_coordinate = jnp.array(converter(args.start_coordinate))
    end_coordinate = jnp.array(converter(args.end_coordinate))
    
    data_loader = DownloadJetStreams(time=args.time,
                                     day=args.day,
                                     month=args.month,
                                     year=args.year,
                                     pressure_level=args.pressure_level,
                                     area = (90, -180, -90, 180),
                                     save_path=args.data_path,
                                     )
    
    lons, lats, u, v = data_loader()
    
    lons, lats, u, v = jnp.array(lons), jnp.array(lats), jnp.array(u), jnp.array(v)
    speed = jnp.sqrt(u**2+v**2)
    
    interpolation_speed = BilinearInterpolation(lons, lats, speed)
    interpolation_u = BilinearInterpolation(lons, lats, u)
    interpolation_v = BilinearInterpolation(lons, lats, v)
    
    lon_fine, lat_fine, speed_fine = interpolation_speed.interpolate_field(grid_size=(800,400))
    _, _, u_fine = interpolation_u.interpolate_field(grid_size=(800,400))
    _, _, v_fine = interpolation_v.interpolate_field(grid_size=(800,400))

    field_fun = lambda t,x: jnp.array([interpolation_u(x[0], x[1]),
                                       interpolation_v(x[0], x[1])])
    
    manifold_jet_stream = WGS84Earth_Chart(force_field=field_fun, 
                          flight_speed=args.flight_speed, 
                          height=args.height,
                          grid_points=args.grid_points,
                          max_iter=args.max_iter,
                          tol=args.tolerance,
                          )
    
    travel_time, flight_path = manifold_jet_stream.geodesic(start_coordinate, end_coordinate)
    
    manifold_no_jet_stream = WGS84Earth_Chart(force_field=None, 
                                              flight_speed=args.flight_speed, 
                                              height=args.height,
                                              grid_points=args.grid_points,
                                              max_iter=args.max_iter,
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
    plot_3d_earth(start_coordinate, end_coordinate, flight_path,
                  start_city=args.start_city, end_city=args.end_city,
                  figure_path=args.figure_path)
    plot_jet_streams(lon_fine, lat_fine, u_fine, v_fine, speed_fine, 
                     flight_path =flight_path,
                     figure_path=args.figure_path,
                     )
    
    
    plot_3d_earth_multi_paths(start_coordinate, end_coordinate, [flight_path, flight_path_no_js, flight_path_euclidean],
                              start_city=args.start_city, end_city=args.end_city,
                              figure_path=args.figure_path)
    
    plot_jet_streams_comparison(lon_fine, lat_fine, u_fine, v_fine, speed_fine, 
# %%
                     flight_paths = {'With Jet Stream': [travel_time, flight_path],

                                    'No Jet Stream': [travel_time_no_js, flight_path_no_js],
                                    'Straight line': [travel_time_euclidean, flight_path_euclidean]},
                     start_end_labels = {'start': args.start_city,
                                         'end': args.end_city,
                                         },
                     figure_path=args.figure_path,
                     title="Optimal Flight Path (250 hPa)"
                     )
    
    return

#%% Main code

def run_no_jet_field(args):
    
    if not os.path.exists(args.figure_path):
        os.makedirs(args.figure_path)
        
    converter = PositionEncoding()
    
    start_coordinate = jnp.array(converter(args.start_coordinate))
    end_coordinate = jnp.array(converter(args.end_coordinate))
    
    manifold = WGS84Earth_Chart(force_field=None, 
                          flight_speed=args.flight_speed, 
                          height=args.height,
                          grid_points=args.grid_points,
                          max_iter=args.max_iter,
                          tol=args.tolerance,
                          )
    
    travel_time, flight_path = manifold.geodesic(start_coordinate, end_coordinate)
    
    plot_earth_chart(start_coordinate, end_coordinate, flight_path,
                     figure_path=args.figure_path)
    plot_3d_earth(start_coordinate, end_coordinate, flight_path,
                  start_city=args.start_city, end_city=args.end_city,
                  figure_path=args.figure_path)
    plot_earth_region_chart(start_coordinate, end_coordinate, flight_path,
                            figure_path=args.figure_path)
    
    return

#%% Run main

if __name__ == "__main__":
    
    args = parse_args()
    
    #import airportsdata

    #airports = airportsdata.load("IATA")
    
    # Get Tokyo Haneda
    #print(airports)
    #hnd = airports['HND']
    
    #print(hnd['name'])
    #print(hnd['lat'], hnd['lon'])

    
    if args.use_jet_stream:
        run_wtih_jet_stream(args)
    else:
        run_no_jet_field(args)