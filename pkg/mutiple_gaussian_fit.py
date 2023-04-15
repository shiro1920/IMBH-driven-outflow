import numpy as np
from scipy import integrate
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from scipy.signal import find_peaks


############################################################################################
# 高斯模型
############################################################################################
def gaussian(x, *p):
    model = p[0] * np.exp(-0.5 * (x - p[1]) ** 2 / p[2] ** 2)
    for i in range(3, len(p), 3):
        amp = p[i]
        mean = p[i + 1]
        sigma = p[i + 2]
        model += amp * np.exp(-0.5 * (x - mean) ** 2 / sigma ** 2)

    return model


############################################################################################
# 计算初始值
############################################################################################
def emline_amp(wave, flux, names, window=10):
    emline_names = ['NII[6548]', 'Ha[6563]', 'NII[6583]', 'Hb[4861]',
                    'OIII[4959]', 'OIII[5007]']
    emline_wave = [6548.03, 6562.8, 6583.41, 4861.3, 4958.92, 5006.84]

    Amp = []
    lc = []
    for i in names:
        idx_i = emline_names.index(i)
        lam_i = emline_wave[idx_i]
        mask_i = np.where(np.abs(wave - lam_i) < window)
        emline_data = flux[mask_i]
        indices = find_peaks(emline_data, height=10, distance=15, prominence=10)
        peak_flux = emline_data[indices[0]]
        if indices[0].size >0:
            Amp.append(peak_flux)
        else:
            Amp.append(1e-6)
        lc.append(lam_i)

    return Amp, lc


############################################################################################
# 根据高斯的参数信息计算动力学信息
############################################################################################
def single_gauss_calculate(wave, spec_error, spec_con, *p, linetype='narrow', vmode=None):
    #### single gauss function
    def make_gauss(amp, mean, sigma):
        s = -1.0 / (2 * sigma * sigma)

        def f(x):
            return amp * np.exp(s * (x - mean) * (x - mean))

        return f

    amp = p[0]
    mean = p[1]
    sigma = p[2]
    fwhm = 2.355 * sigma

    if linetype == 'narrow':
        ### flux & flux error
        flux, integrate_err = integrate.quad(make_gauss(amp, mean, sigma), mean - 3 * fwhm, mean + 3 * fwhm)
        mask = (wave > mean - 3 * fwhm) & (wave < mean + 3 * fwhm)
        flux_err_spec = np.sqrt(np.nansum((spec_error[mask]) ** 2))
        flux_err = np.sqrt(integrate_err ** 2 + flux_err_spec ** 2)
        ### equivalentwidth
        ew = flux / np.median(spec_con)
        ### velocity
        if vmode == None:
            v = np.nan
        if vmode == 'Ha':
            xc = 6562.8
            v = ((mean - xc) / xc) * 3 * 10 ** 5
        if vmode == 'Hb':
            xc = 4861.3
            v = ((mean - xc) / xc) * 3 * 10 ** 5
        if vmode == 'O3a':
            xc = 4958.92
            v = ((mean - xc) / xc) * 3 * 10 ** 5
        if vmode == 'O3b':
            xc = 5006.84
            v = ((mean - xc) / xc) * 3 * 10 ** 5

        ### output
        name = ['flux', 'flux_error', 'velocity', 'equivalentwidth']
        em_data = [flux, flux_err, v, ew]
        emline_data = dict(zip(name, em_data))
        return emline_data

    if linetype == 'broad':
        ### flux & flux error
        flux, integrate_err = integrate.quad(make_gauss(amp, mean, sigma), mean - 3 * fwhm, mean + 1 * fwhm)
        mask = (wave > mean - 3 * fwhm) & (wave < mean + 1 * fwhm)
        flux_err_spec = np.sqrt(np.nansum((spec_error[mask]) ** 2))
        flux_err = np.sqrt(integrate_err ** 2 + flux_err_spec ** 2)
        ### equivalentwidth
        ew = flux / np.median(spec_con)
        ### velocity
        if vmode == None:
            v = np.nan
        if vmode == 'Ha':
            xc = 6562.8
            v = ((mean - xc) / xc) * 3 * 10 ** 5
        if vmode == 'Hb':
            xc = 4861.3
            v = ((mean - xc) / xc) * 3 * 10 ** 5
        if vmode == 'O3a':
            xc = 4958.92
            v50 = ((mean - xc) / xc) * 3 * 10 ** 5
            v80 = (((mean - 1.09 * fwhm) - xc) / xc) * 3 * 10 ** 5
            v_name = ['v50', 'v80']
            v_data = [v50, v80]
            v = dict(zip(v_name, v_data))
        if vmode == 'O3b':
            xc = 5006.84
            v50 = ((mean - xc) / xc) * 3 * 10 ** 5
            v80 = (((mean - 1.09 * fwhm) - xc) / xc) * 3 * 10 ** 5
            v_name = ['v50', 'v80']
            v_data = [v50, v80]
            v = dict(zip(v_name, v_data))
        ### output
        name = ['flux', 'flux_error', 'velocity', 'equivalentwidth']
        em_data = [flux, flux_err, v, ew]
        emline_data = dict(zip(name, em_data))
        return emline_data
    ############################################################################################


############################################################################################
# fit_status 拟合参数信息
############################################################################################
def emline_fit_status(fit_wave, fit_flux, fit_err, para, para_err, mode='Ha'):
    para_error = np.sqrt(np.diag(para_err))
    chi2 = np.sum((fit_flux - gaussian(fit_wave, *para)) ** 2 / fit_err ** 2)
    reduced_chi2 = chi2 / (fit_wave.shape[0] - para.shape[0])
    line_num = len(para) / 3
    fit_para = np.split(para, line_num)
    fit_para_err = np.split(para_error, line_num)

    if mode == 'Ha':
        if line_num == 3:
            names = ['NII[6548]', 'Ha[6563]', 'NII[6583]']
        else:
            names = ['NII[6548]', 'Ha[6563]', 'NII[6583]', 'Ha[6563]_broad']
    if mode == 'Hb':
        if line_num == 1:
            names = ['Hb[4861]']
        else:
            names = ['Hb[4861]', 'Hb[4861]_broad']
    if mode == 'O3a':
        if line_num == 1:
            names = ['OIII[4959]']
        else:
            names = ['OIII[4959]', 'OIII[4959]_broad']
    if mode == 'O3b':
        if line_num == 1:
            names = ['OIII[5007]']
        else:
            names = ['OIII[5007]', 'OIII[5007]_broad']

    line_para = [names[i] + '_para' for i in range(len(names))]
    line_paraerr = [names[i] + '_perr' for i in range(len(names))]
    line_chi2 = ['fit_Chi2', 'fit_rChi2']

    fit_para = dict(zip(line_para, fit_para))
    fit_perr = dict(zip(line_paraerr, fit_para_err))
    fit_chi2 = dict(zip(line_chi2, [chi2, reduced_chi2]))

    return fit_para, fit_perr, fit_chi2


############################################################################################
# fit_flux 拟合发射线的动力学信息
############################################################################################
def emline_fit_properties(wave, spec_error, spec_con, para, mode='Ha'):
    line_num = len(para) / 3
    p_t = np.split(para, line_num)

    flux = []
    flux_err = []
    v = []
    ew = []
    if mode == 'Ha':
        for j in range(0, len(p_t)):
            p_j = p_t[j]
            emline_data = single_gauss_calculate(wave, spec_error, spec_con, *p_j, linetype='narrow', vmode='Ha')
            flux.append(emline_data['flux'])
            flux_err.append(emline_data['flux_error'])
            v.append(emline_data['velocity'])
            ew.append(emline_data['equivalentwidth'])
        if line_num == 3:
            names = ['NII[6548]', 'Ha[6563]', 'NII[6583]']
        else:
            names = ['NII[6548]', 'Ha[6563]', 'NII[6583]', 'Ha[6563]_broad']

    if mode == 'Hb':
        for j in range(0, len(p_t)):
            p_j = p_t[j]
            emline_data = single_gauss_calculate(wave, spec_error, spec_con, *p_j, linetype='narrow', vmode='Hb')
            flux.append(emline_data['flux'])
            flux_err.append(emline_data['flux_error'])
            v.append(emline_data['velocity'])
            ew.append(emline_data['equivalentwidth'])

        if line_num == 1:
            names = ['Hb[4861]']
        else:
            names = ['Hb[4861]', 'Hb[4861]_broad']

    if mode == 'O3a':
        if line_num == 1:
            p_j = p_t[0]
            emline_data = single_gauss_calculate(wave, spec_error, spec_con, *p_j, linetype='narrow', vmode='O3a')
            flux.append(emline_data['flux'])
            flux_err.append(emline_data['flux_error'])
            v.append(emline_data['velocity'])
            ew.append(emline_data['equivalentwidth'])
            names = ['OIII[4959]']
        else:
            p_narrow = p_t[0]
            p_broad = p_t[1]
            narrow_data = single_gauss_calculate(wave, spec_error, spec_con, *p_narrow, linetype='narrow', vmode='O3a')
            broad_data = single_gauss_calculate(wave, spec_error, spec_con, *p_broad, linetype='broad', vmode='O3a')
            flux = [narrow_data['flux'], broad_data['flux']]
            flux_err = [narrow_data['flux_error'], broad_data['flux_error']]
            v = [narrow_data['velocity'], broad_data['velocity']]
            ew = [narrow_data['equivalentwidth'], broad_data['equivalentwidth']]
            names = ['OIII[4959]', 'OIII[4959]_broad']

    if mode == 'O3b':
        if line_num == 1:
            p_j = p_t[0]
            emline_data = single_gauss_calculate(wave, spec_error, spec_con, *p_j, linetype='narrow', vmode='O3b')
            flux.append(emline_data['flux'])
            flux_err.append(emline_data['flux_error'])
            v.append(emline_data['velocity'])
            ew.append(emline_data['equivalentwidth'])
            names = ['OIII[5007]']
        else:
            p_narrow = p_t[0]
            p_broad = p_t[1]
            narrow_data = single_gauss_calculate(wave, spec_error, spec_con, *p_narrow, linetype='narrow', vmode='O3b')
            broad_data = single_gauss_calculate(wave, spec_error, spec_con, *p_broad, linetype='broad', vmode='O3b')
            flux = [narrow_data['flux'], broad_data['flux']]
            flux_err = [narrow_data['flux_error'], broad_data['flux_error']]
            v = [narrow_data['velocity'], broad_data['velocity']]
            ew = [narrow_data['equivalentwidth'], broad_data['equivalentwidth']]
            names = ['OIII[5007]', 'OIII[5007]_broad']

    if mode == 'S2':
        for j in range(0, len(p_t)):
            p_j = p_t[j]
            emline_data = single_gauss_calculate(wave, spec_error, spec_con, *p_j, linetype='narrow', vmode='S2')
            flux.append(emline_data['flux'])
            flux_err.append(emline_data['flux_error'])
            v.append(emline_data['velocity'])
            ew.append(emline_data['equivalentwidth'])
        names = ['SII[6717]', 'SII[6731]']

    ######## output
    line_flux = [names[i] + '_flux' for i in range(len(names))]
    line_err = [names[i] + '_fluxerr' for i in range(len(names))]
    line_v = [names[i] + '_velocity' for i in range(len(names))]
    line_ew = [names[i] + '_EW' for i in range(len(names))]

    emline_flux = dict(zip(line_flux, flux))
    emline_flux_err = dict(zip(line_err, flux_err))
    emline_v = dict(zip(line_v, v))
    emline_ew = dict(zip(line_ew, ew))

    return emline_flux, emline_flux_err, emline_v, emline_ew


############################################################################################
# 用gaussian拟合单根发射线
############################################################################################
def single_line_fit(wave, galaxy, noise, continum, maxfev=10000, mode='Hb'):
    #################    set Initial value    ############################
    if mode == 'Hb':
        name = ['Hb[4861]']
    if mode == 'O3a':
        name = ['OIII[4959]']
    if mode == 'O3b':
        name = ['OIII[5007]']

    Amp, lc = emline_amp(wave, galaxy, name, window=10)
    mask = np.where(np.abs(wave - lc[0] < 20))
    fit_wave = wave[mask]
    fit_flux = galaxy[mask]
    fit_err = noise[mask]
    fit_con = continum[mask]

    ################    narrow line    ############################
    p0_n = np.array([Amp[0], lc[0], 1.])
    p0_n_bounds = ([0., lc[0] - 5, 0.],
                   [Amp[0], lc[0] + 5, 2.])
    popt_n, pcov_n = curve_fit(gaussian, fit_wave, fit_flux, p0=p0_n, bounds=p0_n_bounds,
                               sigma=fit_err, absolute_sigma=True, maxfev=maxfev)
    fit_para_n, fit_perr_n, fit_chi2_n = emline_fit_status(fit_wave, fit_flux, fit_err, popt_n, pcov_n, mode=mode)

    ###################    broad line    #############################
    p0_b = np.array([Amp[0], lc[0], 1.,
                     Amp[0], lc[0], 3.])
    p0_b_bounds = ([0, lc[0] - 5, 0, 0, lc[0] - 5, 2],
                   [Amp[0], lc[0] + 5, 2, Amp[0], lc[0] + 5, 5])
    popt_b, pcov_b = curve_fit(gaussian, fit_wave, fit_flux, p0=p0_b, bounds=p0_b_bounds,
                               sigma=fit_err, absolute_sigma=True, maxfev=maxfev)
    fit_para_b, fit_perr_b, fit_chi2_b = emline_fit_status(fit_wave, fit_flux, fit_err, popt_b, pcov_b, mode=mode)

    diff_chi2 = fit_chi2_n['fit_Chi2'] - fit_chi2_b['fit_Chi2']
    if diff_chi2 < 0.21 * fit_wave.shape[0]:
        popt = popt_n
        pcov = pcov_n
        fit_para, fit_perr, fit_chi2 = emline_fit_status(fit_wave, fit_flux, fit_err, popt, pcov, mode=mode)
        emline_flux, emline_flux_err, emline_v, emline_ew = emline_fit_properties(wave, noise, fit_con, popt, mode=mode)
    else:
        popt = popt_b
        pcov = pcov_b
        fit_para, fit_perr, fit_chi2 = emline_fit_status(fit_wave, fit_flux, fit_err, popt, pcov, mode=mode)
        emline_flux, emline_flux_err, emline_v, emline_ew = emline_fit_properties(wave, noise, fit_con, popt, mode=mode)

    return name, fit_para, fit_perr, fit_chi2, emline_flux, emline_flux_err, emline_v, emline_ew


class emline_fit():
    """
    for multiple gaussian profile
    """

    def __init__(self, wave, flux, error, continum=None, pix_y=None, pix_x=None,
                 maxfev=10000, plot=True, quiet=False):

        self.wave = wave
        self.py = pix_y
        self.px = pix_x
        self.maxfev = maxfev
        self.plot = plot
        self.quiet = quiet

        if flux.shape != error.shape:
            print("flux and error must be same shape")

        if continum is None:
            self.continum = np.full_like(flux,-99999)
        else:
            if len(continum.shape) == 1:
                self.continum = continum
            else:
                self.continum = continum[:,pix_y, pix_x]

        if len(flux.shape) == 1 & len(error.shape) == 1:
            self.galaxy = flux
            self.noise = error
        else:
            if pix_y is None or pix_x is None:
                print("IFU data need image coordinates (x,y)")
            else:
                self.galaxy = flux[:, pix_y, pix_x]
                self.noise = error[:, pix_y, pix_x]
        self.npix = self.galaxy.shape[0]

        assert np.all((error > 0) & np.isfinite(error)), \
            "error must be a positive vector"
        assert np.all(np.isfinite(flux)), "flux must be finite value"
        assert np.all(np.isfinite(self.continum)), "continum must be finite value"

        if maxfev < 0:
            print("maxfev must be a postive value")

        self.fit_para = {}
        self.fit_para_error = {}
        self.flux = {}
        self.flux_error = {}
        self.velocity = {}
        self.equivalentwidth = {}

        self.Ha_fit()
        self.Hb_fit()

        if not quiet:
            self.output()
        if plot:
            self.showimg()



    def Ha_fit(self):

        #################    set Initial value    ############################
        haname = ['NII[6548]', 'Ha[6563]', 'NII[6583]']
        Amp, lc = emline_amp(self.wave, self.galaxy, haname, window=10)
        mask_Ha = ((self.wave > 6530) & (self.wave < 6610))
        fit_wave = self.wave[mask_Ha]
        fit_spec = self.galaxy[mask_Ha]
        fit_err = self.noise[mask_Ha]
        fit_con = self.continum[mask_Ha]
        ################    narrow line    ############################
        p0 = np.array([Amp[0], lc[0], 1.,
                       Amp[1], lc[1], 1.,
                       Amp[2], lc[2], 1.])
        p0_bounds = ([0, 6543, 0, 0, 6558.5, 0, 0, 6578, 0],
                     [Amp[0], 6553, 2, Amp[1], 6567.5, 3, Amp[2], 6588, 2])
        popt_n, pcov_n = curve_fit(gaussian, fit_wave, fit_spec, p0=p0, bounds=p0_bounds,
                                   sigma=fit_err, absolute_sigma=True, maxfev=self.maxfev)
        fit_para_n, fit_perr_n, fit_chi2_n = emline_fit_status(fit_wave, fit_spec, fit_err, popt_n, pcov_n, mode='Ha')
        ###################    broad line    #############################
        p0_b = np.array([Amp[0], lc[0], 1.,
                         Amp[1], lc[1], 1.,
                         Amp[2], lc[2], 1.,
                         Amp[1], lc[1], 3.])
        p0_b_bounds = ([0, 6543, 0, 0, 6557, 0, 0, 6578, 0, 0, 6557, 2],
                       [Amp[0], 6553, 2, Amp[1], 6567, 3, Amp[2], 6588, 2, Amp[1], 6567, 5])
        popt_b, pcov_b = curve_fit(gaussian, fit_wave, fit_spec, p0=p0_b, bounds=p0_b_bounds,
                                   sigma=fit_err, absolute_sigma=True, maxfev=self.maxfev)
        fit_para_b, fit_perr_b, fit_chi2_b = emline_fit_status(fit_wave, fit_spec, fit_err, popt_b, pcov_b, mode='Ha')
        ######################    compare fit results ##########################
        diff_chi2 = fit_chi2_n['fit_Chi2'] - fit_chi2_b['fit_Chi2']
        if diff_chi2 < 14:
            popt = popt_n
            pcov = pcov_n
            ha_para, ha_perr, ha_chi2 = emline_fit_status(fit_wave, fit_spec, fit_err, popt, pcov, mode='Ha')
            ha_flux, ha_flux_err, ha_v, ha_ew = emline_fit_properties(self.wave, self.noise, fit_con, popt, mode='Ha')
        else:
            popt = popt_b
            pcov = pcov_b
            ha_para, ha_perr, ha_chi2 = emline_fit_status(fit_wave, fit_spec, fit_err, popt, pcov, mode='Ha')
            ha_flux, ha_flux_err, ha_v, ha_ew = emline_fit_properties(self.wave, self.noise, fit_con, popt, mode='Ha')

        self.fit_para.update(ha_para)
        self.fit_para_error.update(ha_perr)
        self.flux.update(ha_flux)
        self.flux_error.update(ha_flux_err)
        self.velocity.update(ha_v)
        self.equivalentwidth.update(ha_ew)
        self.Ha_Chi2 = ha_chi2
        pass

    def Hb_fit(self):

        hbname, hb_para, hb_perr, hb_chi2, hb_flux, hb_flux_err, hb_v, hb_ew =\
            single_line_fit(self.wave, self.galaxy, self.noise, self.continum, maxfev=10000, mode='Hb')
        self.fit_para.update(hb_para)
        self.fit_para_error.update(hb_perr)
        self.flux.update(hb_flux)
        self.flux_error.update(hb_flux_err)
        self.velocity.update(hb_v)
        self.equivalentwidth.update(hb_ew)

        O3aname, O3a_para, O3a_perr, O3a_chi2, O3a_flux, O3a_flux_err, O3a_v, O3a_ew = \
            single_line_fit(self.wave, self.galaxy, self.noise, self.continum, maxfev=10000, mode='O3a')
        self.fit_para.update(O3a_para)
        self.fit_para_error.update(O3a_perr)
        self.flux.update(O3a_flux)
        self.flux_error.update(O3a_flux_err)
        self.velocity.update(hb_v)
        self.equivalentwidth.update(O3a_ew)

        O3bname, O3b_para, O3b_perr, O3b_chi2, O3b_flux, O3b_flux_err, O3b_v, O3b_ew = \
            single_line_fit(self.wave, self.galaxy, self.noise, self.continum, maxfev=10000, mode='O3b')
        self.fit_para.update(O3b_para)
        self.fit_para_error.update(O3b_perr)
        self.flux.update(O3b_flux)
        self.flux_error.update(O3b_flux_err)
        self.velocity.update(O3b_v)
        self.equivalentwidth.update(O3b_ew)

        Chi2_hb = hb_chi2['fit_Chi2'] + O3a_chi2['fit_Chi2'] + O3b_chi2['fit_Chi2']
        rChi2_hb = Chi2_hb/(len(hb_para)+len(O3a_para)+len(O3b_para))
        self.Hb_Chi2 = dict(zip(['fit_Chi2','fit_rChi2'],[Chi2_hb,rChi2_hb]))

        pass
    def output(self):
        print('=========================================')
        print('     name        flux      flux error')
        print('=========================================')
        self.gas_name = [key[:-5] for key in self.flux]
        gas_flux = [self.flux[key] for key in self.flux]
        gas_flux_error = [self.flux_error[key] for key in self.flux_error]

        for j, lnames in enumerate(self.gas_name):
            print("%12s %#10.5g  %#8.4g" %
                  (self.gas_name[j], gas_flux[j],
                   gas_flux_error[j]))
        print('==========================================')

    def showimg(self):
        fig = plt.figure(figsize=(12,10))
        ax1 = fig.add_subplot(212)
        mask_Ha   = ((self.wave > 6530) & (self.wave < 6610))
        Ha_names  = [key for key in self.fit_para.keys() if 'NII' in key or 'Ha' in key]
        Ha_paras  = np.array([self.fit_para[k] for k in Ha_names]).flatten()
        ha_wave   = self.wave[mask_Ha]
        fake_Hax  = np.arange(ha_wave[0],ha_wave[-1],0.01)
        ax1.plot(ha_wave, self.galaxy[mask_Ha] * 10 ** (-2), color='k', lw=0.7, alpha=1, label='observed')
        ax1.plot(fake_Hax, gaussian(fake_Hax, *Ha_paras[0:9]) * 10 ** (-2), color='C0', label='narrow-part')
        try:
            ax1.plot(fake_Hax, gaussian(fake_Hax, *Ha_paras[9:12]) * 10 ** (-2), label='broad-part', color='magenta')
        except:
            pass
        ax1.plot(fake_Hax, gaussian(fake_Hax, *Ha_paras) * 10 ** (-2), color='darkorange', lw=2, label='best-fit')

        ax2 = fig.add_subplot(211)
        mask_Hb  = ((self.wave > 4830) & (self.wave < 5040))
        Hb_names = [key for key in self.fit_para.keys() if 'OIII' in key or 'Hb' in key]
        Hb_para  = np.array([self.fit_para[k] for k in Hb_names]).flatten()
        hbn_name = [i for i in Hb_names if 'broad' not in i]
        hbb_name = [i for i in Hb_names if 'broad' in i]
        Hbn_para = np.array([self.fit_para[k] for k in hbn_name]).flatten()
        Hbb_para = np.array([self.fit_para[k] for k in hbb_name]).flatten()
        hb_wave  = self.wave[mask_Hb]
        fake_Hbx = np.arange(hb_wave[0], hb_wave[-1], 0.01)
        ax2.plot(hb_wave, self.galaxy[mask_Hb] * 10 ** (-2), color='k', lw=0.7, alpha=1, label='observed')
        ax2.plot(fake_Hbx, gaussian(fake_Hbx, *Hbn_para) * 10 ** (-2), color='C0', label='narrow-part')
        try:
            ax2.plot(fake_Hbx, gaussian(fake_Hbx, *Hbb_para) * 10 ** (-2), label='broad-part', color='magenta')
        except:
            pass
        ax2.plot(fake_Hbx, gaussian(fake_Hbx, *Hb_para) * 10 ** (-2), color='darkorange', lw=2, label='best-fit')

        ax1.set_xlabel('Rest Wavelength', fontsize=20)
        ax1.set_ylabel(r'${\rm F_\lambda \ [10^{-18}erg/s/cm^2/\AA]}$', fontsize=15)
        ax2.set_ylabel(r'${\rm F_\lambda \ [10^{-18}erg/s/cm^2/\AA]}$', fontsize=15)
        ax2.set_title('Spaxel ' + str(self.py) + '-' + str(self.px) + ' ' + ' fit results', fontsize=15)
        ax1.tick_params(direction='in', labelsize=15, length=5, width=1.0)
        ax2.tick_params(direction='in', labelsize=15, length=5, width=1.0)
        
        plt.legend(fontsize=15.0, markerscale=2)
