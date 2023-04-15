import numpy as np
from PyAstronomy import pyasl

def dgis_CaII(air = True):
#outputs: the list of emission lines
    
    bright  = {'CaII_8500':  8500.36, 'CaII_8544':  8544.44, \
               'CaII_8664':  8664.52}


    outstr = bright

            
    tags   = list(outstr.keys())
    ab     = list(outstr.values())

    # return lines in vacuum by default
    if air == False:
        ab  =pyasl.airtovac2(ab)

    tags = np.array(tags)
    ab   = np.array(ab)
    return outstr, tags, ab
 
