import numpy as np

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
