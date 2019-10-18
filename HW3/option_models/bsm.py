    # -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 22:56:58 2017

@author: jaehyuk
"""

import numpy as np
import scipy.stats as ss
import scipy.optimize as sopt

def price(strike, spot, texp, vol, intr=0.0, divr=0.0, cp_sign=1):
    div_fac = np.exp(-texp*divr)
    disc_fac = np.exp(-texp*intr)
    forward = spot / disc_fac * div_fac

    if( texp<=0 ):
        return disc_fac * np.fmax( cp_sign*(forward-strike), 0 )
    
    # floor vol_std above a very small number
    vol_std = np.fmax(vol*np.sqrt(texp), 1e-32)
    
    d1 = np.log(forward/strike)/vol_std + 0.5*vol_std
    d2 = d1 - vol_std

    price = cp_sign * disc_fac \
        * ( forward * ss.norm.cdf(cp_sign*d1) - strike * ss.norm.cdf(cp_sign*d2) )
    return price

class Model:
    texp, vol, intr, divr = None, None, None, None
    
    def __init__(self, texp, vol, intr=0, divr=0):
        self.texp = texp
        self.vol = vol
        self.intr = intr
        self.divr = divr
    
    def price(self, strike, spot, texp=None, vol=None, cp_sign=1):
        # pas vol and texp if you don't want to use values stored in class
        vol = self.vol if(vol is None) else vol
        texp = self.texp if(texp is None) else texp
        return price(strike, spot, texp, vol, intr=self.intr, divr=self.divr, cp_sign=cp_sign)
    
    def delta(self, strike, spot, texp=None, vol=None, cp_sign=1):
        delta=np.exp(-texp*self.divr)*ss.norm.cdf(np.log((spot/np.exp(-texp*self.intr)*np.exp(-texp*self.divr))/strike)/(self.vol*np.sqrt(texp))+1/2*(self.vol*np.sqrt(texp))*cp_sign)*cp_sign
        return delta

    def vega(self, strike, spot, texp=None, vol=None, cp_sign=1):
        vega=spot*np.exp(-texp*self.divr)*np.sqrt(texp)*ss.norm.pdf(np.log((spot/np.exp(-texp*self.intr)*np.exp(-texp*self.divr))/strike)/(self.vol*np.sqrt(texp))+1/2*(self.vol*np.sqrt(texp)))
        return vega
    
    def gamma(self, strike, spot, texp=None, vol=None, cp_sign=1):
        gamma=gamma=np.exp(-texp*self.divr)*ss.norm.pdf(np.log((spot/np.exp(-texp*self.intr)*np.exp(-texp*self.divr))/strike)/(self.vol*np.sqrt(texp))+1/2*(self.vol*np.sqrt(texp)))/(spot*self.vol*np.sqrt(texp))
        return gamma

    def impvol(self, price_in, strike, spot, texp=None, cp_sign=1):
        texp = self.texp if(texp is None) else texp
        div_fac = np.exp(-texp*self.divr)
        disc_fac = np.exp(-texp*self.intr)
        forward = spot/disc_fac*div_fac
        
        int_val = disc_fac*np.fmax(cp_sign*(forward-strike), 0)
        if(int_val > price_in):
            raise ValueError('Option value is lower than intrinsic value', price_in, int_val) 

        iv_func = lambda _vol: \
            price(strike, spot, texp, _vol, self.intr, self.divr, cp_sign) - price_in
        vol = sopt.brentq(iv_func, 0, 10)
        return vol
