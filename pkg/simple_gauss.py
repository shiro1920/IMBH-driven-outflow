#
# Create gaussian: g = A * exp(-(x - x0)^2/(2*sigma^2))
# P[0] = Flux (sqrt(2*PI)*A*sigma)  (in erg/s/cm^2) 
# P[1] = offset_velocity  (in km/s)  
# P[2] = sigma (in km/s)
# inst_res = instrumental resolution (in km/s)
# dp = d(yval)/d(p[i]) partial derivatives
#
# REVISION HISTORY: 
#    2007-Sept-18 Written by Christy Tremonti (tremonti@as.arizona.edu)
#-
import numpy as np

def simple_gauss(p, fjac=None, x=None, y=None, err=None, inst_res=None, line_ctr=None):

    
    nline =len(line_ctr)
    if inst_res == None:
        inst_res = np.zeros(nline)
    else:
        inst_res  = np.full(nline, inst_res)
    
    cspeed = 2.99792e5
    
    #add gaussian for each line
    yval = 0.0
    for ii in np.arange(nline):
        x0        = line_ctr[ii] * (1 + p[0]/cspeed)
        sigma_kms = np.sqrt(p[1]**2 + inst_res[ii]**2) 
        sigma_a   = sigma_kms/cspeed * x0  
    
        term1     = np.exp( - (x - x0)**2 / (2. * sigma_a**2) )
        g         = p[ii+2] / (np.sqrt(2.*np.pi) * sigma_a) * term1
        yval      = yval + g
    
    bad   = np.isinf(yval)
    n_bad = np.count_nonzero(bad)
    if n_bad > 0:
        yval[bad] = 0.0
    
    status =0
    return([status, (yval-y)/err])
    
