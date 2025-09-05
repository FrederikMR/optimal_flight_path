#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 21:26:11 2025

@author: frederikmobiusrygaard
"""

#%% Modules

import jax.numpy as jnp
from jax import (
    grad, 
    jacfwd, 
    vmap, 
    lax,
    )

from jax import Array
from typing import Tuple

from typing import (
    Callable, 
    Dict,
    )

from abc import (
    ABC, 
    abstractmethod,
    )

from .metrics import Metric
from .line_search import Backtracking

#%% GEORCE Estimation of Tack Points and Geodesics

class ForceFieldGeodesic(ABC):
    def __init__(self,
                 metric:Metric,
                 init_fun:Callable[[Array, Array, int], Array]=None,
                 T:int=1000,
                 tol:float=1e-4,
                 max_iter:int=1000,
                 line_search_params:Dict = {'rho': 0.5},
                 )->None:
        
        self.metric = metric
        self.T = T
        self.tol = tol
        self.max_iter = max_iter
        self.line_search_params = line_search_params
        self.Lt = grad(self.metric.metric, argnums=0)
        self.Lz = jacfwd(self.metric.metric, argnums=1)
        self.Lu = jacfwd(self.metric.metric, argnums=2)
        
        if init_fun is None:
            self.init_fun = lambda z0, zT, T: (zT-z0)*jnp.linspace(0.0,
                                                                   1.0,
                                                                   T,
                                                                   endpoint=False,
                                                                   dtype=z0.dtype)[1:].reshape(-1,1)+z0
        else:
            self.init_fun = init_fun
        
    def __str__(self)->str:
        
        return "Geodesic Computation Object using Control Problem"
    
    def energy(self,
               zs:Array, 
               *args
               )->Array:
        
        us = jnp.vstack((zs[0]-self.z0,
                         zs[1:]-zs[:-1],
                         self.zT-zs[-1]
                         ))
        ts = self.update_ts(zs, us)
        
        val1 = self.metric.metric(self.t0, self.z0, us[0])**2
        val2 = vmap(lambda t,x,v: self.metric.metric(t,x,v)**2)(ts[:-1], zs, us[1:])

        return val1+jnp.sum(val2)
    
    def Denergy(self,
                zs:Array,
                )->Array:
        
        return grad(self.energy)(zs)
    
    def inner_product(self,
                      ts:Array,
                      zs:Array,
                      us:Array,
                      )->Array:
        
        Gs = vmap(self.metric.fundamental_tensor)(ts,zs,us)
        
        return jnp.sum(jnp.einsum('ti,tij,tj->t', us, Gs, us))
    
    def inner_product_h(self,
                        ts:Array,
                        zs:Array,
                        u0:Array,
                        us:Array,
                        )->Array:
        
        Gs = vmap(self.metric.fundamental_tensor)(ts,zs,us)
        
        return jnp.sum(jnp.einsum('ti,tij,tj->t', u0, Gs, u0))
    
    def Ginv(self,
             t:Array,
             z:Array,
             v:Array,
             )->Array:
        
        return jnp.linalg.inv(self.metric.fundamental_tensor(t, z, v))
    
    def rs(self,
           ts:Array,
           zs:Array,
           us:Array,
           )->Array:
        
        return grad(self.inner_product, argnums=0)(ts,zs,us)
    
    def gs(self,
           ts:Array,
           zs:Array,
           us:Array,
           )->Array:
        
        return grad(self.inner_product, argnums=1)(ts,zs,us)
    
    def hs(self,
           ts:Array,
           zs:Array,
           us:Array,
           )->Array:

        return grad(self.inner_product_h, argnums=3)(ts,zs,us,us)
    
    def update_ts(self,
                  zs:Array,
                  us:Array,
                  )->Array:
        
        def step(t:Array,
                 step:Tuple[Array,Array],
                 )->Array:
            
            z, dz = step
            
            t += self.metric.metric(t, z, dz)
            
            return (t,)*2
        
        zs = jnp.vstack((self.z0, zs))
        
        _, ts = lax.scan(step,
                         init=self.t0,
                         xs = (zs, us),
                         )
        
        return ts
    
    def update_zs(self,
                  zs:Array,
                  alpha:Array,
                  us_hat:Array,
                  us:Array,
                  )->Array:
        
        return self.z0+jnp.cumsum(alpha*us_hat[:-1]+(1-alpha)*us[:-1], axis=0)
    
    def update_us(self,
                  gs_inv:Array,
                  mus:Array,
                  )->Array:
        
        return -0.5*jnp.einsum('tij,tj->ti', gs_inv, mus)
    
    def pi(self,
           rs:Array,
           Lts:Array,
           )->Array:
        
        def step(pis:Array,
                 step:Tuple[Array,Array],
                 )->Tuple[Array, Array]:
            
            rs, Ls = step
            
            return ((rs+pis*Ls+pis),)*2
        
        _, pi = lax.scan(step,
                         xs=(rs[::-1], Lts[::-1]),
                         init=0.0,
                         )
        
        return jnp.hstack((pi[::-1], 0.0)).reshape(-1,1)
    
    def unconstrained_opt(self, 
                          rs:Array,
                          hs:Array, 
                          gs:Array, 
                          gs_inv:Array,
                          pis:Array,
                          Lts:Array,
                          Lzs:Array,
                          Lus:Array,
                          )->Array:

        g_cumsum = jnp.vstack((jnp.cumsum((gs+pis[1:]*Lzs)[::-1], axis=0)[::-1], jnp.zeros((1,self.dim))))
        ginv_sum = jnp.sum(gs_inv, axis=0)
        rhs = jnp.sum(jnp.einsum('tij,tj->ti', gs_inv, g_cumsum+hs+pis*Lus), axis=0)+2.0*self.diff
        #lhs = -jnp.linalg.inv(ginv_sum)
        #muT = jnp.einsum('ij,j->i', lhs, rhs)
        muT = -jnp.linalg.solve(ginv_sum, rhs)
        mus = muT+g_cumsum+hs+pis*Lus
        
        return mus
    
    def cond_fun(self, 
                 carry:Tuple[Array,Array,Array, Array, int],
                 )->Array:
        
        ts, zs, us, rs, hs, gs, gs_inv, grad, idx = carry
        
        norm_grad = self.T*jnp.linalg.norm(grad.reshape(-1))

        return (norm_grad>self.tol) & (idx < self.max_iter)
    
    def georce_step(self,
                    carry:Tuple[Array,Array,Array, Array, int],
                    )->Array:
        
        ts, zs, us, rs, hs, gs, gs_inv, grad, idx = carry
        
        Lts = vmap(self.Lt, in_axes=(0,0,0))(ts[:-1],zs,us[1:])
        Lzs = vmap(self.Lz, in_axes=(0,0,0))(ts[:-1],zs,us[1:])
        Lus = jnp.vstack((self.Lu(self.t0, self.z0, us[0]).reshape(-1,self.dim),
                          vmap(self.Lu, in_axes=(0,0,0))(ts[:-1],zs,us[1:]),
                          ))
        pis = self.pi(rs, Lts)
        
        mus = self.unconstrained_opt(rs, hs, gs, gs_inv,
                                     pis, Lts, Lzs, Lus)
        
        us_hat = self.update_us(gs_inv, mus)

        tau = self.line_search(zs, us_hat, us)

        us = tau*us_hat+(1.-tau)*us
        zs = self.z0+jnp.cumsum(us[:-1], axis=0)
        ts = self.update_ts(zs, us)
        
        rs = self.rs(ts[:-1], zs, us[1:])
        hs = self.hs(jnp.hstack((self.t0, ts[:-1])), jnp.vstack((self.z0.reshape(1,-1), zs)), us)
        gs = self.gs(ts[:-1], zs, us[1:])
        gs_inv = jnp.vstack((self.Ginv(self.t0, self.z0, us[0]).reshape(-1,self.dim,self.dim), 
                             vmap(self.Ginv)(ts[:-1],zs,us[1:])))
        grad = self.Denergy(zs)
        
        return (ts, zs, us, rs, hs, gs, gs_inv, grad, idx+1)
    
    def __call__(self, 
                 t0:Array,
                 z0:Array,
                 zT:Array,
                 )->Array:

        self.t0 = t0
        self.z0 = z0
        self.zT = zT
        self.diff = zT-z0
        
        dtype = z0.dtype
        self.dim = len(z0)
        
        self.line_search = Backtracking(obj_fun=self.energy,
                                        update_fun=self.update_zs,
                                        grad_fun = lambda z,*args: self.Denergy(z).reshape(-1),
                                        **self.line_search_params,
                                        )
        
        zs = self.init_fun(z0,zT,self.T)
        us = jnp.ones((self.T, self.dim), dtype=dtype)*self.diff/self.T
        ts = self.update_ts(zs, us)

        rs = self.rs(ts[:-1], zs, us[1:])
        hs = self.hs(jnp.hstack((self.t0, ts[:-1])), jnp.vstack((self.z0.reshape(1,-1), zs)), us)
        gs = self.gs(ts[:-1], zs, us[1:])
        gs_inv = jnp.vstack((self.Ginv(self.t0,self.z0, us[0]).reshape(-1,self.dim,self.dim), 
                             vmap(self.Ginv)(ts[:-1],zs,us[1:])))
        grad = self.Denergy(zs)
        
        ts, zs, us, rs, hs, gs, gs_inv, grad, idx = lax.while_loop(self.cond_fun, 
                                                                   self.georce_step, 
                                                                   init_val=(ts, zs, us, rs, hs, gs, gs_inv, grad, 0),
                                                                   )
        
        zs = jnp.vstack((z0, zs, zT))
        ts = jnp.hstack((t0, ts))
            
        return ts, zs, grad, idx
