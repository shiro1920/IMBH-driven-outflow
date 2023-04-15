import glob
from time import perf_counter as clock
from os import path
import ppxf as ppxf_package
from ppxf.ppxf import ppxf
import ppxf.ppxf_util as util

import numpy as np
from astropy.io import fits

def single_pixel_ppxf_fit(rest_wave, spec_flux, error, degree, plot, formal_error):
    ppxf_dir = path.dirname(path.realpath(util.__file__))

    z = 0   # redshift 
    wavelength = rest_wave
    mask = (wavelength > 4800) & (wavelength < 7409)

    flux = spec_flux[mask]
    galaxy = flux/np.median(flux)   
    lam_gal = wavelength[mask]
    noise = error[mask]     
    c = 299792.458                  
    frac = lam_gal[1]/lam_gal[0]    
    fwhm = np.full_like(wavelength, 2.42, dtype=np.double)
    fwhm_gal = fwhm[mask]           
    velscale = np.log(frac)*c       

    #######################################################????#########################################################################

    vazdekis = glob.glob(ppxf_dir + '/miles_models/Eun1.30Z*.fits')
    fwhm_tem = 2.51 # Vazdekis+10 spectra have a constant resolution FWHM of 2.51A.
    hdu = fits.open(vazdekis[0])
    ssp = hdu[0].data
    h2 = hdu[0].header
    lam_temp = h2['CRVAL1'] + h2['CDELT1']*np.arange(h2['NAXIS1'])
    lamRange_temp = [np.min(lam_temp), np.max(lam_temp)]
    sspNew = util.log_rebin(lamRange_temp, ssp, velscale=velscale)[0]
    templates = np.empty((sspNew.size, len(vazdekis)))
    fwhm_gal = np.interp(lam_temp, lam_gal, fwhm_gal)

    fwhm_dif = np.sqrt((fwhm_gal**2 - fwhm_tem**2).clip(0))
    sigma = fwhm_dif/2.355/h2['CDELT1'] # Sigma difference in pixels

    for j, fname in enumerate(vazdekis):
        hdu = fits.open(fname)
        ssp = hdu[0].data
        ssp = util.gaussian_filter1d(ssp, sigma)  # perform convolution with variable sigma
        sspNew = util.log_rebin(lamRange_temp, ssp, velscale=velscale)[0]
        templates[:, j] = sspNew/np.median(sspNew) # Normalizes templates

    c = 299792.458   # km/s
    dv = c*np.log(lam_temp[0]/lam_gal[0])  # eq.(8) of Cappellari (2017)
    goodpixels = util.determine_goodpixels(np.log(lam_gal), lamRange_temp, z)

    vel = c*np.log(1 + z)   # eq.(8) of Cappellari (2017)
    start = [vel, 100.]  # (km/s), starting guess for [V, sigma]
    t = clock()

    pp = ppxf(templates, galaxy, noise, velscale, start,
              goodpixels=goodpixels, plot=plot, moments=4,
              degree=degree, vsyst=dv, clean=False, lam=lam_gal)
    
    if formal_error:
        print("Formal errors:")
        print("     dV    dsigma   dh3      dh4")
        print("".join("%8.2g" % f for f in pp.error*np.sqrt(pp.chi2)))


        print('Elapsed time in PPXF: %.2f s' % (clock() - t))
        
    return pp