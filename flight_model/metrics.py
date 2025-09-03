#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 18:21:17 2025

@author: frederikmobiusrygaard
"""

#%% Modules

import jax.numpy as jnp

from jax import Array
from typing import (
    Callable, 
    )

from abc import (
    ABC, 
    abstractmethod,
    )

#%% Metric

class Metric(ABC):
    @abstractmethod
    def fundamental_tensor(self,
                           t:float,
                           z:Array,
                           v:Array,
                           )->Array:
        
        return NotImplemented
    
    @abstractmethod
    def metric(self,
               t:float,
               z:Array,
               v:Array,
               )->Array:
        
        return NotImplemented
    
#%% Earht Metric

class ForceFieldMetric(Metric):
    def __init__(self,
                 riemannian_metric:Callable[[Array], Array],
                 force_field:Callable[[float,Array],Array],
                 speed:float=.25, #Object speed in km/s (default: 900 km/h = .25 km/s)
                 )->None:
        
        self.riemannian_metric = riemannian_metric
        self.force_field = force_field
        self.speed = speed
        
        return
    
    def __str__(self,)->str:
        
        return "Time-dependent Finsler metric for the earth with a force field"
    
    def fundamental_tensor(self,
                           t:Array,
                           z:Array,
                           v:Array,
                           )->Array:
        
        g = self.riemannian_metric(z)
        force = self.force_field(t,z)
        
        lam = 1./((self.speed**2)-jnp.einsum('ij,i,j->', g, force, force))
        f = jnp.dot(g, force)
        
        a = g*lam+jnp.einsum('i,j->ij', f, f)*(lam**2)
        b = -f*lam
        
        inner = jnp.sqrt(jnp.einsum('ij,i,j->', a, v, v))
        l = jnp.dot(a, v)/inner
        
        gv = (1.0+jnp.dot(b, v)/inner)*(a-jnp.einsum('i,j->ij', l, l))\
            +jnp.einsum('i,j->ij', b+l, b+l)
        
        return 0.5*gv
    
    def metric(self,
               t:Array,
               z:Array,
               v:Array,
               )->Array:
        
        g = self.riemannian_metric(z)
        force = self.force_field(t,z)
        
        lam = 1./((self.speed**2)-jnp.einsum('ij,i,j->', g, force, force))
        f = jnp.dot(g, force)
        
        a = g*lam+jnp.einsum('i,j->ij', f, f)*(lam**2)
        b = -f*lam
        
        term1 = jnp.einsum('ij,i,j->', a, v, v)
        term2 = jnp.dot(b, v)
        
        return jnp.sqrt(term1)+term2