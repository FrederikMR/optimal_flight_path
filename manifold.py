#%% Modules

import jax.numpy as jnp
from jax import (
    vmap,
    jacfwd,
    )

from jax import Array
from typing import (
    Callable,
    Tuple, 
    Optional,
    )

from abc import (
    ABC,
    abstractmethod,
    )

from metrics import ForceFieldMetric
from geodesics import ForceFieldGeodesic

#%% Base Earth Class

class Earth(ABC):
    @abstractmethod
    def geodesic(self,
                 z1:Array,
                 z2:Array,
                 )->Array:
        
        return NotImplemented
    

#%% WGS84Earth model

class WGS84Earth(Earth):
    """
    WGS84 Earth ellipsoid model.
    
    Parameters:
        half_axes: Tuple of semi-axes in kilometers, defaults to WGS84:
                   (a=6378.137, a=6378.137, c=6356.752)
    """
    def __init__(self,
                 force_field:Optional[Callable[[float,Array],Array]]=None,
                 height:float=11.0,
                 half_axes:Optional[Tuple[float,float,float]]=(6_378.137,
                                                               6_378.137,
                                                               6_356.752), #Measured in km
                 flight_speed:float=900., #Speed of plane
                 grid_points:int=100,
                 max_iter:int=1_000,
                 tol:float=1e-2,
                 )->None:
        super().__init__()
        
        if half_axes is None:
            self.half_axes = jnp.array((1.,1.,1.)) + height
        else:
            self.half_axes = jnp.array(half_axes) + height
        
        if force_field is None:
            self.force_field = lambda t,x: jnp.zeros_like(x, dtype=x.dtype)
        else:
            self.force_field = (
                lambda t,x: force_field(t,
                                        self.ecef_to_geodetic(self._steo_to_earth(x)))/jnp.max(self.half_axes)
                )
            
        self.normalized_half_axes = self.half_axes / jnp.max(self.half_axes)
        
        self.flight_speed = flight_speed
        self.normalized_flight_speed = flight_speed/jnp.max(self.half_axes)
        
        self.metric = ForceFieldMetric(self._riemannian_metric, 
                                       self.force_field,
                                       speed=self.normalized_flight_speed,
                                       )
        self.geodesic_solver = ForceFieldGeodesic(metric=self.metric,
                                                  T=grid_points,
                                                  max_iter=max_iter,
                                                  tol=tol,
                                                  )
        
        return
    
    def __str__(self)->str:
        
        return "WGS84 Earth model"
    
    def earth_to_map(self,
                     x:Array,
                     )->Array:
        
        x /= self.half_axes
        
        lon = jnp.arctan2(x.T[1], x.T[0])
        lat = jnp.arcsin(x.T[-1])
        
        lon *= 180./jnp.pi #to degrees
        lat *= 180./jnp.pi#to degrees
        
        return jnp.stack((lon, lat)).T
    
    def map_to_earth(self,
                     z:Array,
                     )->Array:
        
        lon, lat = z.T
        
        #lon *= jnp.pi/180. #to radians
        #lat *= jnp.pi/180. #to radians
        
        x1 = jnp.cos(lon)*jnp.cos(lat)
        x2 = jnp.sin(lon)*jnp.cos(lat)
        x3 = jnp.sin(lat)
        
        return jnp.stack((x1.T,x2.T,x3.T)).T*self.half_axes
    
    def _steo_to_normalized_earth(self,
                                  z:Array,
                                  )->Array:
        
        X,Y = z
        
        x = 2*X / ((X**2)+(Y**2)+1.)
        y = 2*Y / ((X**2)+(Y**2)+1.)
        z = ((X**2)+(Y**2)-1.)/((X**2)+(Y**2)+1.)
        
        return jnp.stack((x,y,z))*self.normalized_half_axes
    
    def _steo_to_earth(self,
                       z:Array,
                       )->Array:
        
        X,Y = z
        
        x = 2*X / ((X**2)+(Y**2)+1.)
        y = 2*Y / ((X**2)+(Y**2)+1.)
        z = ((X**2)+(Y**2)-1.)/((X**2)+(Y**2)+1.)
        
        return jnp.stack((x,y,z))*self.half_axes

    def _earth_to_steo(self,
                       x:Array,
                       )->Array:
        
        X,Y,Z = x/self.half_axes
        
        x = X/(1.-Z)
        y = Y/(1.-Z)
        
        return jnp.stack((x,y))
    
    def _riemannian_metric(self,
                           z:Array,
                           )->Array:

        Jf = jacfwd(self._steo_to_normalized_earth)(z)
        
        return jnp.einsum('...ki,...kj->...ij', Jf, Jf)
    
    def geodetic_to_ecef(self,
                         x:Array,
                         h: float = 0.0,
                         )->Array:
        
        lon_deg, lat_deg = x
        
        # Custom ellipsoid parameters (in meters)
        a = self.half_axes[0]  # semi-major axis
        b = self.half_axes[1]  # semi-minor axis
    
        # Eccentricity squared
        e2 = 1 - (b**2 / a**2)
    
        # Convert latitude and longitude to radians
        lat = jnp.deg2rad(lat_deg)
        lon = jnp.deg2rad(lon_deg)
    
        # Radius of curvature in the prime vertical
        N = a / jnp.sqrt(1 - e2 * jnp.sin(lat)**2)
    
        # Compute ECEF coordinates
        x = (N + h) * jnp.cos(lat) * jnp.cos(lon)
        y = (N + h) * jnp.cos(lat) * jnp.sin(lon)
        z = (N * (1 - e2) + h) * jnp.sin(lat)
    
        return jnp.stack((x, y, z))
    
    def ecef_to_geodetic(self, 
                         x:Array,
                         )->Array:
        
        x,y,z = x
        
        # Custom ellipsoid parameters
        a = self.half_axes[0]   # semi-major axis in meters
        b = self.half_axes[1]   # semi-minor axis in meters
        e2 = 1 - (b**2 / a**2)  # eccentricity squared
        ep2 = (a**2 - b**2) / b**2  # second eccentricity squared
    
        # Longitude
        lon = jnp.arctan2(y, x)
    
        # Compute preliminary values
        p = jnp.sqrt(x**2 + y**2)
        theta = jnp.arctan2(z * a, p * b)
    
        # Latitude (Bowring's formula)
        sin_theta = jnp.sin(theta)
        cos_theta = jnp.cos(theta)
    
        lat = jnp.arctan2(z + ep2 * b * sin_theta**3,
                         p - e2 * a * cos_theta**3)
    
        # Radius of curvature in the prime vertical
        #N = a / jnp.sqrt(1 - e2 * jnp.sin(lat)**2)
    
        # Altitude above ellipsoid
        #h = p / jnp.cos(lat) - N
    
        # Convert radians to degrees
        lat_deg = jnp.rad2deg(lat)
        lon_deg = jnp.rad2deg(lon)
    
        return jnp.stack((lon_deg, lat_deg)) #h
    
    def geodesic(self,
                 z1:Array, #Points in (longitude, lattitude)
                 z2:Array, #Points in (longitude, lattitude)
                 )->Tuple[Array,Array]:
        
        x1 = self.geodetic_to_ecef(z1)
        x2 = self.geodetic_to_ecef(z2)
        
        z1 = self._earth_to_steo(x1)
        z2 = self._earth_to_steo(x2)
        
        val = self.geodesic_solver(0.0,z1,z2)
        curve = val[1]
        travel_time = val[0][-1]
        
        xcurve = vmap(self._steo_to_earth)(curve)
        zcurve = vmap(self.ecef_to_geodetic)(xcurve)
        
        return travel_time, zcurve
    