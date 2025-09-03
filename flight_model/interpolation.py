#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 20:07:29 2025

@author: frederikmobiusrygaard
"""

#%% Modules

import jax.numpy as jnp
from jax import vmap

from jax import Array
from typing import Tuple

from abc import ABC

#%% Bilinear interpolation

class BilinearInterpolation(ABC):
    def __init__(self,
                 x:Array, #2d grid 
                 y:Array,
                 values:Array, #values for 2d grid
                 )->None:
        
        self.x = x
        self.y = y
        self.values = values
        
        return
    
    def __str__(self,)->str:
        
        return "Bilinear interpolation object"
    
    def interpolate_field(self,
                          grid_size:Tuple[int,int]=(800, 400),
                          ):
        """
        Use JAX bilinear interpolation to resample `field` on a finer grid.
        """
        lon_min, lon_max = self.x.min(), self.x.max()
        lat_min, lat_max = self.y.min(), self.y.max()

        lon_fine = jnp.linspace(lon_min, lon_max, grid_size[0])
        lat_fine = jnp.linspace(lat_min, lat_max, grid_size[1])
        lon_grid, lat_grid = jnp.meshgrid(lon_fine, lat_fine)

        # Vectorize interpolation function over grid points
        interp_func = vmap(vmap(
            lambda xi, yi: self.interpolate_points(xi, yi),
            in_axes=(0, None)),  # vectorize over xi
            in_axes=(None, 0))   # then vectorize over yi

        field_smooth = interp_func(lon_fine, lat_fine)
        return lon_fine, lat_fine, field_smooth
    
    def interpolate_points(self,
                           xi:Array,
                           yi:Array,
                           )->Array:
        
        x,y = self.x, self.y
        
        # Find indices
        ix = jnp.clip(jnp.searchsorted(x, xi) - 1, 0, len(x) - 2)
        iy = jnp.clip(jnp.searchsorted(y, yi) - 1, 0, len(y) - 2)

        x0 = x[ix]
        x1 = x[ix + 1]
        y0 = y[iy]
        y1 = y[iy + 1]

        f00 = self.values[iy, ix]
        f10 = self.values[iy, ix + 1]
        f01 = self.values[iy + 1, ix]
        f11 = self.values[iy + 1, ix + 1]

        # Normalize distances
        tx = (xi - x0) / (x1 - x0)
        ty = (yi - y0) / (y1 - y0)

        # Bilinear interpolation formula
        interp_val = (1 - tx) * (1 - ty) * f00 + \
                     tx * (1 - ty) * f10 + \
                     (1 - tx) * ty * f01 + \
                     tx * ty * f11

        return interp_val
    
    def __call__(self,
                 xi:Array,
                 yi:Array,
                 )->Array:
        
        return self.interpolate_points(xi, yi)