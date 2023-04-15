import pandas as pd
import numpy as np
from astropy.io import fits 

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches4
from matplotlib import colors
import matplotlib.patches as mpatches

import scipy.constants as C
import scipy.interpolate as spi

from itertools import product

import warnings
from scipy.signal import find_peaks

from scipy import integrate
from scipy.optimize import curve_fit

from plotbin.display_pixels import display_pixels

import random

def gauss_doublet(x,*p):
    f = np.zeros_like(x)
    for i in np.arange(0,np.asarray(p).shape[0],4):
        f_O3b = p[i]
        f_O3a = f_O3b/3
        mean_a = p[i+1]
        mean_b = p[i+2]
        sigma_a = p[i+3]
        sigma_b = sigma_a 
        f+= ((f_O3a/(sigma_a*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mean_a)/sigma_a)**2)+
             (f_O3b/(sigma_b*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mean_b)/sigma_b)**2))
    return f 

def CHI2(fitwave,fitF,fitFerr,*p,fitmodel):
    
    CHI2 = np.sum(((fitF-fitmodel(fitwave,*p))/fitFerr)**2)
    rCHI2 = CHI2/(fitwave.shape[0]-np.asarray(p).shape[0])
    
    return CHI2,rCHI2

def gas_kine(p,perr):
    
    p = np.asarray(p)
    perr = np.asarray(perr)
    p_err = np.sqrt(np.diag(perr))
    OIII_xc = 5006.84
    
    if p.shape[0] == 4:
        O3b_flux = p[0]
        O3b_flux_err = p_err[0]
        O3a_flux = O3b_flux/3
        O3a_flux_err = O3b_flux_err/3
        O3a_mean = p[1]
        O3b_mean = p[2]
        O3b_sigma = O3a_sigma = p[3]
        
        v_narrow = ((O3b_mean-OIII_xc)/OIII_xc)*3*10**5/0.8718
        
        line_name = ['[OIII]4959_flux','[OIII]4959_flux_err',
                     '[OIII]5007_flux','[OIII]5007_flux_err',
                     '[OIII]5007_v']
        gas_kin = [O3a_flux,O3a_flux_err,O3b_flux,O3b_flux_err,v_narrow]
        
        
    else:
        O3b_flux = p[0]
        O3b_flux_err = p_err[0]
        O3a_flux = O3b_flux/3
        O3a_flux_err = O3b_flux_err/3
        O3a_mean = p[1]
        O3b_mean = p[2]
        O3b_sigma = O3a_sigma = p[3]
        
        O3b_flux_o = p[4]
        O3b_flux_o_err = p_err[4]
        O3a_flux_o = O3b_flux_o/3
        O3a_flux_o_err = O3b_flux_o_err/3
        O3a_mean_o = p[5]
        O3b_mean_o = p[6]
        O3b_sigma_o = O3a_sigma_o = p[7]
        
        v_narrow = (((O3b_mean-OIII_xc)/OIII_xc)*3*10**5)/0.8718
        v_broad = (((O3b_mean_o-OIII_xc)/OIII_xc)*3*10**5)/0.8718
        fwhm = 2.355*O3b_sigma_o
        W80 = 1.09 * (fwhm/OIII_xc)*3*10**5
        v_0 = v_broad-v_narrow
        v_outf = -v_0+(W80/2)
        
        line_name = ['[OIII]4959_flux','[OIII]4959_flux_err',
                     '[OIII]5007_flux','[OIII]5007_flux_err',
                     '[OIII]5007_v',
                     '[OIII]4959_broad_flux','[OIII]4959_broad_flux_err',
                     '[OIII]5007_broad_flux','[OIII]5007_broad_flux_err',
                     '[OIII]5007_broad_v','[OIII]5007_broad_W80',
                     '[OIII]5007_broad_vout',]
        gas_kin = [O3a_flux,O3a_flux_err,O3b_flux,O3b_flux_err,v_narrow,
                  O3a_flux_o,O3a_flux_o_err,O3b_flux_o,O3b_flux_o_err,v_broad,W80,v_outf]
    
    gas_kinemics = dict(zip(line_name,gas_kin))
    
    return gas_kinemics

def O3_doublet_fit(wave,spec_flux,spec_error,plot,savefig):
    
    galaxy_wave = wave
    galaxy_flux = spec_flux
    galaxy_err = spec_error
    
    mask_OIII = ((galaxy_wave > 4940) & (galaxy_wave < 5030))
    fitOIII_wave = galaxy_wave[mask_OIII]
    fitOIII_flux = galaxy_flux[mask_OIII]
    fitOIII_err = galaxy_err[mask_OIII]

    mask_O3a = ((galaxy_wave > 4950) & (galaxy_wave < 4970))
    mask_O3b = ((galaxy_wave > 4997) & (galaxy_wave < 5017))
    flux_iO3b = np.abs(np.sum(galaxy_flux[mask_O3b])) 
    flux_iO3a = flux_iO3b/3
    
    
    #####  doublet  without outflow
    p_O3n = np.array([flux_iO3b,4959.,5007.,1.])
    p_O3n_bounds =([-flux_iO3b,4954.,5004.,0.],
                    [flux_iO3b,4964.,5010.,2.])
    
    poptO3n,pcovO3n = curve_fit(gauss_doublet,fitOIII_wave,fitOIII_flux,
                                      p0=p_O3n,bounds=p_O3n_bounds,
                                      sigma=fitOIII_err, absolute_sigma=True,
                                      maxfev=100000)
    chi2_n,rchi2_n = CHI2(fitOIII_wave,fitOIII_flux,fitOIII_err,*poptO3n,fitmodel=gauss_doublet)
    
    #####  doublet  with  outflow
    p_O3o = np.array([flux_iO3b,4959.,5007.,1.,
                       flux_iO3b,4959.,5007.,3.])
    p_O3o_bounds =([-flux_iO3b,4954.,5004.,0.,-flux_iO3b,4954.,5004.,2.],
                    [flux_iO3b,4964.,5010.,2., flux_iO3b,4964.,5010.,5.])
    
    poptO3o,pcovO3o = curve_fit(gauss_doublet,fitOIII_wave,fitOIII_flux,
                                      p0=p_O3o,bounds=p_O3o_bounds,
                                      sigma=fitOIII_err, absolute_sigma=True,
                                      maxfev=100000)
    chi2_o,rchi2_o = CHI2(fitOIII_wave,fitOIII_flux,fitOIII_err,*poptO3o,fitmodel=gauss_doublet)
    
    if chi2_n-chi2_o >17:
        poptOIII = poptO3o
        pcovOIII = pcovO3o
        chi2 = chi2_o
        rchi2 = rchi2_o
    else:
        poptOIII = poptO3n
        pcovOIII = pcovO3n
        chi2 = chi2_n
        rchi2 = rchi2_n
        
    gas_kinemics = gas_kine(poptOIII,pcovOIII)
    status_name = ["fit_parameter","para_error","CHI2","rCHI2"]
    status = [poptOIII,np.sqrt(np.diag(pcovOIII)),chi2,rchi2]
    fit_status = dict(zip(status_name,status))
        
    
    if plot:
        fig = plt.figure(figsize=(10,6))
        fake_hbx = np.arange(4940,5030,0.01)

        ax1 = fig.add_subplot(111)
        ax1.plot(fitOIII_wave,gauss_doublet(fitOIII_wave,*poptOIII)*10**(-2),color='darkorange',lw=2,label='best-fit')
        ax1.plot(fitOIII_wave,fitOIII_flux*10**(-2),color='k', lw=0.7, alpha=1,label='observed')
        ax1.set_xlabel('Rest Wavelength')
        ax1.set_ylabel(r'${\rm Flux\ [10^{-18}erg/s/cm^2/\AA]}$')
        plt.legend()

        ax1.plot(fake_hbx,gauss_doublet(fake_hbx,*poptOIII[0:4])*10**(-2),label='narrow-part',color='C0',lw=1)
        try:
            ax1.plot(fake_hbx,gauss_doublet(fake_hbx,*poptOIII[4:8])*10**(-2),label='broad-part',color='magenta',lw=1)
        except:
            pass

        plt.legend()
        plt.show()
         
        if savefig:
            plt.savefig('O3_doublet_fit.pdf',dpi = 600)
    return fit_status,gas_kinemics