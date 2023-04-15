import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from astropy.io import fits
from astropy import wcs, coordinates
import astropy.units as u
from astropy.cosmology import LambdaCDM

class MUSE(object):
    """
    read spectrum data from muse fits file, and do simple reduce
    """

    def __init__(self, muse_source, z=0):
        """
        read data from MUSE fits file
        """
        ###############################  const  #############################################
        self.COSMO = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)
        self.ESP = 1e-8
        self.C = 299792  # speed of light (km/s)
        #####################################################################################
        self.hdr = muse_source[1].header
        self.RA = muse_source[0].header['RA']
        self.DEC = muse_source[0].header['DEC']
        self.wcs_cube = wcs.WCS(self.hdr)                                   # WCS data
        self.z = z
        self.dist_L = self.COSMO.luminosity_distance(self.z)                  # distance
        self.arc2pix = np.abs(self.hdr['CD1_1']*3600.0)                     # arcsec/pixel
        self.pc2pix = self.arc2pix/2.06e5*self.dist_L.to(u.pc)               # pc/pixel
        self.flux_unit = self.hdr['BUNIT']                                  # flux density unit
        self.wavelength = np.arange(self.hdr['CRVAL3'],
                           self.hdr['CRVAL3'] + self.hdr['CD3_3'] * self.hdr['NAXIS3'],
                           self.hdr['CD3_3'])                           # wavelength in unit AA
        self.wave_rest = self.wavelength / (1 + self.z)                          # rest frame wavelength
        self.IFU_flux = muse_source[1].data                                 # flux density data
        self.IFU_noise = np.sqrt(muse_source[2].data)                       # flux density error
        self.datacube_size = self.IFU_flux.shape
        self.ysize = self.datacube_size[1]
        self.xsize = self.datacube_size[2]
        self.image_file = None

    def plot_spec(self, x, y, ax=None, showImage=True,
                  wave_min=None, wave_max=None, flux_min=None, flux_max=None):
        """
        show spectrum plot
        """
        flux_density = self.IFU_flux[:, y, x]
        wavelength = self.wave_rest

        if ax == None:
            fig = plt.figure(figsize=(8, 5))
            ax = fig.add_subplot(111)

        ax.plot(wavelength, flux_density)
        plt.ylim(flux_min, flux_max)
        plt.xlim(wave_min, wave_max)
        plt.tick_params(direction='in', labelsize=15, length=5, width=1.0)
        ax.set_ylabel(r'${\rm Flux\ [10^{-18}erg/s/cm^2/\AA]}$', fontsize=20)
        ax.set_xlabel('Rest Wavelength', fontsize=20)

        if showImage:
            plt.show()

    def RGB_image(self, ax=None, showImage=True):
        """
        show rgb-image
        """
        try:
            imagedata = mpimg.imread(self.image_file)
        except:
            print("{} image file doesn't exist!".format(self.plateifu))
            imagedata = np.zeros((2, 2))

        if ax == None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        ax.imshow(imagedata)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        # set the 'show=False' when use it as a figure modifier
        if showImage:
            plt.show()

    def nan_mask(self, wave_min=4800, wave_max=7400):
        """
        mask nan data
        """
        flux = self.IFU_flux
        wave = self.wave_rest
        ppxf_mask = ((wave > wave_min) & (wave < wave_max))
        nan_mask = np.zeros((self.ysize, self.xsize))

        for y in range(self.ysize):
            for x in range(self.xsize):
                isnan = np.isnan(flux[:, y, x][ppxf_mask])
                if np.any(isnan) == True:
                    nan_mask[y, x] = 0
                else:
                    nan_mask[y, x] = 1

        return nan_mask