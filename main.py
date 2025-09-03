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
    WGS84Earth,
    )

from flight_model.plotting import (
    plot_earth_chart,
    plot_3d_earth,
    plot_earth_region_chart,
    plot_jet_streams,
    )

#%% Argeparse

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
    parser.add_argument('--flight_speed', default=800.,
                        type=float)
    parser.add_argument('--time', default='12:00',
                        type=str)
    parser.add_argument('--day', default='15',
                        type=str)
    parser.add_argument('--month', default='08',
                        type=str)
    parser.add_argument('--year', default='2025',
                        type=str)
    parser.add_argument('--pressure_level', default='250',
                        type=str)
    parser.add_argument('--use_jet_stream', default=True,
                        type=bool)
    parser.add_argument('--grid_points', default=100,
                        type=int)
    parser.add_argument('--max_iter', default=1000,
                        type=int)
    parser.add_argument('--tolerance', default=1e-2,
                        type=float)
    parser.add_argument('--figure_path', default='figures_output/',
                        type=str)
    parser.add_argument('--data_path', default='data_output/',
                        type=str)

    args = parser.parse_args()
    return args

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
    
    manifold = WGS84Earth(force_field=field_fun, 
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
    plot_jet_streams(lon_fine, lat_fine, u_fine, v_fine, speed_fine, flight_path,
                     figure_path=args.figure_path)
    
    return

#%% Main code

def run_no_jet_field(args):
    
    if not os.path.exists(args.figure_path):
        os.makedirs(args.figure_path)
        
    converter = PositionEncoding()
    
    start_coordinate = jnp.array(converter(args.start_coordinate))
    end_coordinate = jnp.array(converter(args.end_coordinate))
    
    manifold = WGS84Earth(force_field=None, 
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
    
    if args.use_jet_stream:
        run_wtih_jet_stream(args)
    else:
        run_no_jet_field(args)