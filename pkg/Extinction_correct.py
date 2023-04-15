def Extinction_correct(F_Ha, F_Hb, F_obs=None,
                       lam_obs=None, gal_type="AGN", R_V=None):
    """
    do extiction correct for single emmision line, or deredden for single spectrum.
    INPUT:
        F_Ha,F_Hb     : array like; Ha and Hb flux for estimate E(B-V) 
        F_obs, lam_obs: array like; observed flux, flue error and line center wave or spectrum and wavelength vector 
        gal_type      : str; "AGN" or "SF", different galaxy type refer to different intrinsic Ha/Hb ratio, default = "AGN";
                        Kewley et al. (2006) apply 3.1 for AGN and 2.86 for star-forming galaxies 
        R_V           : ratio of total to selective extinction, default = 4.05; 
                        Calzetti et al. (2000) estimate R_V = 4.05 +/- 0.80 from optical-IR observations 
    OUTPUT:
        E_BV          : array like, same shape as F_Ha and F_hb; color excess map 
        F_int         : array like, same shape as F_obs; intrisic flux or deredden spectrum  
             
    """
    # ---------------------------------------------
    import numpy as np
    import warnings
    warnings.filterwarnings("ignore")
    # ---------------------------------------------
    
    # k(λ) for different wavelength; and default R_v = 4.05; 
    # from (Calzetti 2000);
    
    R_V = 4.05 if R_V is None else R_V
    if lam_obs is None:
        print("no input emission line flux and wavelength")
        pass
    else:
        k_lam = np.full_like(lam_obs,np.nan)
        if 912. < lam_obs < 6300.:
            k_lam = 2.659 * (-2.156 + 1.509/lam_obs - 0.198/lam_obs**2 + 0.011/lam_obs**3) + R_V
        if 6300. < lam_obs < 22000.:
            k_lam = 2.659 * (-1.857 + 1.040/lam_obs) + R_V
    
        
    # estimate E(B-V)
    # different galaxy type refer to different intrinsic Ha/Hb ratio; 
    # from (Kewley et al. 2006); 
    if gal_type == "AGN":
        ratio_int = 3.1
    if gal_type == "SF":
        ratio_int = 2.86
    if F_Ha.shape != F_Hb.shape:
        print("Ha and Hb flux array must be same shape!")
        
    E_BV = np.full_like(F_Ha,np.nan)
    E_BV = 1.97 * np.log10((F_Ha/F_Hb)/ratio_int)
    E_BV[E_BV < 0] = np.nan
    
    # do extinction correct for emline flux 
    if F_obs is None:
        F_int = np.nan
    else:
        F_int = np.full_like(F_obs,np.nan)
        F_int = F_obs * 10**(0.4*E_BV*k_lam)
    
    return E_BV, F_int 
