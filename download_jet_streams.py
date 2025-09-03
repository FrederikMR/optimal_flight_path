#%% Modules

import numpy as np

import cdsapi
import xarray as xr

from pathlib import Path

from numpy.typing import ArrayLike
from typing import Tuple

from abc import ABC


#%% Download Jet Streams

class DownloadJetStreams(ABC):
    def __init__(self,
                 time:str='12:00',
                 day:str='15',
                 month:str='08',
                 year:str='2025',
                 pressure_level:str='250',
                 area:Tuple[int,int,int,int] = (90, -180, -90, 180),
                 save_path:str='',
                 )->None:
        
        self.time = time
        self.day = day
        self.month = month
        self.year = year
        
        self.pressure_level = pressure_level
        self.area = area
        
        self.save_path = (
            ''.join((save_path,
                     f'jet_stream_uv_{self.pressure_level}_{self.time}_{self.day}_{self.month}_{self.year}.nc',
                     ),
                    )
            )
        
        return
    
    def __str__(self,)->str:
        
        return "Downloader object that downloads jet stream data from Copernicus"
    
    # Step 2: Load data
    def load_data(self,
                  )->Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
        
        ds = xr.open_dataset(self.save_path)
        u = ds['u'].squeeze()
        v = ds['v'].squeeze()
        lons = ds['longitude'].values
        lats = ds['latitude'].values
        return lons, lats, u.values, v.values

    # Helper to ensure strictly increasing arrays
    def make_strictly_increasing(self,
                                 arr:ArrayLike,
                                 )->ArrayLike:
        
        arr = np.array(arr)
        for i in range(1, len(arr)):
            if arr[i] <= arr[i-1]:
                arr[i] = arr[i-1] + 1e-5  # tiny bump to enforce strict increase
        return arr

    # Step 3: Sort and ensure strict monotonicity of coordinates and data
    def ensure_strictly_increasing(self,
                                   lons:ArrayLike, 
                                   lats:ArrayLike, 
                                   u:ArrayLike, 
                                   v:ArrayLike,
                                   )->Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
        # Convert longitudes from [0, 360) to [-180, 180)
        lons = (lons + 180) % 360 - 180

        # Sort longitudes and reorder u, v accordingly
        if not np.all(np.diff(lons) > 0):
            idx_lon = np.argsort(lons)
            lons = lons[idx_lon]
            u = u[:, idx_lon]
            v = v[:, idx_lon]

        # Sort latitudes and reorder u, v accordingly
        if not np.all(np.diff(lats) > 0):
            idx_lat = np.argsort(lats)
            lats = lats[idx_lat]
            u = u[idx_lat, :]
            v = v[idx_lat, :]

        # Enforce strictly increasing (handle duplicates)
        lons = self.make_strictly_increasing(lons)
        lats = self.make_strictly_increasing(lats)

        return lons, lats, u, v
    
    def download_jet_stream_data(self,
                                 )->None:
        
        c = cdsapi.Client()
        c.retrieve(
            'reanalysis-era5-pressure-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': ['u_component_of_wind', 'v_component_of_wind'],
                'pressure_level': self.pressure_level,
                'year': self.year,
                'month': self.month,
                'day': self.day,
                'time': self.time,
                'area': self.area,  # global
            },
            self.save_path,
        )
        
        return
    
    def __call__(self,
                 )->Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
        
        data_path = Path(self.save_path)
        
        if not data_path.exists():
            self.download_jet_stream_data()
            if not data_path.exists():
                raise ValueError("The data cannot be downloaded")
            
        lons, lats, u, v = self.load_data()
        lons, lats, u, v = self.ensure_strictly_increasing(lons, lats, u, v)
        
        u *= 3.6 #convert to km/h
        v *= 3.6 #convert to km/h
        
        return lons, lats, u, v
        
            
        
        
        
        
        
        
        