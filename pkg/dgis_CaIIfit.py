#+
# NAME:
#   emlinefit
#
# PURPOSE:
#   Fit emission lines in a continuum subtracted spectrum.  The
#   velocity offset of the emission lines from the systemic redshift is
#   assumed to be the same for all lines. The line width (in km/s) is
#   also assumed to be the same.  To achieve an optimal fit, it is desirable
#   to know the wavelength dependence of the instrumental resolution.
#
# CALLING SEQUENCE:
#    s = semlinefit(wave, flux, continuum, err, inst_res=, yfit=, 
#                   /struct_only)
#
# INPUTS:
#    wave: rest wavelength array [npix]
#    flux: flux array (not continuum subtracted!) [npix]
#    continuum: continuum fit [npix]
#    err: error in fluxes [npix]
#
# OPTIONAL INPUTS:
#    vinst_res: instrumental resolution in km/s.  It should either be a 
#              scalar or an nline array (instrumental res at each line 
#              center). The default value is 75 km/s. 
#    struct_only: when set returns an empty output structure then exits
#
# OUTPUTS:
#    s:  structure containing v_off (velocity offset in km/s), sigma
#        (line width in km/s) and flux, flux_err, and EW for each
#        line. The EW is simply the line flux divided by the smoothed 
#        continuum at line center (positive=emission).
#
# NOTES:  
#    Fluxes may be positive or negative.  This is necessary to insure
#    that noise isn't systematically fit as weak emission.  A flux
#    error of -1 indicates that the line has been masked out from
#    the fit.  Errors of 0.0, mean that the fitting program decided
#    to drop the line from the fit.  This also occurs in cases where
#    the lines are tied, for example [NII] 6548 (tied to 6584).   
#
#    Fitting is done by MPFIT
#    http://cow.physics.wisc.edu/%7Ecraigm/idl/idl.html
#
# REVISION HISTORY: 
#    2007-Sept-18 Written by Christy Tremonti (tremonti@as.arizona.edu)
#    2009-Feb-01 Added plotting
#
#-----------------------------------------------------------------------------
from importlib import reload
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter

import dgis_CaII
reload(dgis_CaII)
from dgis_CaII import dgis_CaII

import simple_gauss
reload(simple_gauss)
from simple_gauss import simple_gauss

import mpfit
reload(mpfit)
from mpfit import mpfit

def dgis_CaIIfit(wave=None, flux=None, continuum=None, err=None,  struct_only = False, noplot=True): #vinst_res=75,

    # tie line width and velocity offset of *all* lines
    linestr, line_names, linewl = dgis_CaII() #line_structure, line_names, line_center_wavelength
    nline = len(line_names)

    # Convert dispersion to line centers
    #vinst_res= np.squeeze([vinst_res])
    #if vinst_res.size == 1:
    #        line_res = np.zeros(nline) + vinst_res
    #if vinst_res.size > 1:
    #        line_res = np.interp(linewl, wave, vinst_res)
    
    #-----------------------------------------------------------
    #Make output structure
    c       = 299792.458  # speed of light in km/s
    
    ans = {'v_off': 0.0, 'v_off_err': 0.0, 'sigma': 0.0, 'sigma_err': 0.0}
    for ii in np.arange(nline):
        new_ans = {line_names[ii] + '_flux': 0.0, line_names[ii] + '_err': 0.0, line_names[ii] + '_ew': 0.0}
        ans.update(new_ans)
    if struct_only == True:
        return ans
    #-------------------------------------------------------------
    # Set up input structure for fitting
    starting_ampl = np.interp(linewl, wave, flux)
    

    parinfo = [{'value':0., 'fixed':0, 'limited':[0,0], 'limits':[0.,0.], 'tied': ''} for i in np.arange(nline+2)]

    parinfo[0]['value']   = -1.0 #center
    parinfo[0]['limited'] = [1,1]
    parinfo[0]['limits']  = [-600.0,600.0]
    parinfo[1]['value']   = 20.0 #sigma
    parinfo[1]['limited'] = [1, 1]
    parinfo[1]['limits']  = [0.0, 300.0]
    for i in np.arange(2, nline+2):
        parinfo[i]['value']  = starting_ampl[i-2] * 6 # line flux

    
        
   
    #------------------------------------------------------------------
    # Clip to include just the spectrum around the lines (This speeds things up a lot)

    wave_clip = np.full(len(wave), 0)
    dz        = 1500.0 / c # clipping interval

    for ii in np.arange(nline):
        ok = (wave > linewl[ii]*(1-dz)) & (wave < linewl[ii]*(1+dz)) & \
             (err > 0) & ~np.isinf(err) & ~np.isinf(flux)
        num_pix = np.count_nonzero(ok)
        
        if (num_pix > 10):
            wave_clip[ok] = 1
        else: 
            # Don't fit masked lines
            parinfo[2 + ii]['fixed'] = 1
            parinfo[2 + ii]['value'] = 0.0

    igood = (wave_clip == 1)
    
    ngood = np.count_nonzero(igood)
    if ngood < 10:
        return ans
 
    #-----------------------------------------------------------------
    # Do the fit!
    continsub = flux - continuum

   
    ##set inst_res=0 to get the observed line width

    
    fa = {'x': wave[igood], 'y': continsub[igood], 'err': err[igood], 'inst_res': 0.0, 'line_ctr': linewl}
    p0 = np.zeros(nline+2)
    for i in np.arange(nline+2):
        p0[i] = parinfo[i]['value']

    
    mp_res  = mpfit(simple_gauss, p0, parinfo=parinfo, functkw=fa, quiet=1)
    #print('EMLINE FIT STATUS:', mp_res.status)
    #print('EMLINE FIT NITER:', mp_res.niter)

    
    params               = mp_res.params
    perror               = mp_res.perror
    
    
    status_tmp, chi_root = simple_gauss(params, x=wave, y=continsub, err=err, line_ctr=linewl)
    
    yfit                 = chi_root*err+continsub

    #-----------------------------------------------
    # Parse output 
    ans['v_off']     = params[0] #km/s
    ans['v_off_err'] = perror[0]
    ans['sigma']     = params[1] #km/s
    ans['sigma_err'] = perror[1]

    # measure continuum at line center
    scont     = savgol_filter(continuum, 31, 3)
    line_cont = np.interp(linewl, wave, scont)
    ##store fluxes and errors
    ans_tags = list(ans.keys())
    
    for ii in np.arange(nline):
        
        if  (linewl[ii] > max(wave)) or ( linewl[ii] < min(wave) ):
            ans[line_names[ii]+'_flux_err'] = -1  # assign an negative error
            continue        
        ans[line_names[ii]+'_flux'] =  params[ii+2]
        ans[line_names[ii]+'_err']  =  perror[ii+2]
        ans[line_names[ii]+'_ew']   =  params[ii+2]/line_cont[ii]
    
    #############################plot
    if noplot == False:
        fig = plt.figure(figsize=(12, 7))
        matplotlib.rc('xtick', labelsize=30)
        matplotlib.rc('ytick', labelsize=30)

        ax = fig.add_subplot(1,1,1)
        ax.plot(wave, continsub)
        igood_win = (igood)      
        ax.plot(wave[igood_win], yfit[igood_win])
                        #print(line_names[ii], snr)
    #return mp_res, yfit
    
    return ans
