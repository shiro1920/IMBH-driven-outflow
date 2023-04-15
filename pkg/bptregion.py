def bptregion(x, y, mode='N2'):
    '''
    Args:
        lines: dictionary contains all the needed for bpt
            ['Hb-4862', 'OIII-5008','Ha-6564','NII-6585']
      x: log10(NII/Ha) or log10(SII/Ha) or log10(OI/Ha)
      y: log10(OIII/Hb)
      mode: mode "N2" -> x = log10(NII/Ha)
            mode "S2" -> x = log10(SII/Ha)
            mode "O1" -> x = log10(OI/Ha)  ! not surpport yet
    Note:
      x, y should be masked array, 
      example: x = np.ma.array(x)
    '''
    # check starforming, composite or AGN
    # region = np.zeros_like(lines[0].data)
    from numpy.ma import is_masked
    if mode == 'N2':
        ke01 = 0.61/(x-0.47)+1.19
        ka03 = 0.61/(x-0.05)+1.3
        schawinski_line = 1.05*x+0.45
        region_AGN = np.logical_or(np.logical_and(x<0.47, y>ke01), x>0.47)
        region_composite = np.logical_and(y<ke01, y>ka03)
        region_starforming = np.logical_and(x<0.05, y<ka03)
        # depleted
        #region_seyfert = np.logical_and(x>np.log10(0.6), y>np.log10(3.))
        #region_liner = np.logical_and(region_AGN, np.logical_and(x>np.log10(0.6), y<np.log10(3.)))
        # adapted from Schawinski2007
#         region_seyfert = np.logical_and(region_AGN, y>schawinski_line)
#         region_liner = np.logical_and(region_AGN, y<schawinski_line)
        if is_masked(x) or is_masked(y):
            return region_AGN.filled(False), region_composite.filled(False), region_starforming.filled(False), region_seyfert.filled(False), region_liner.filled(False)
        else:
            return region_AGN, region_composite, region_starforming

    if mode == 'S2':
        ke01_line = 0.72/(x-0.32)+1.3
        seyfert_liner_line = 1.89*x+0.76
        region_seyfert = np.logical_and(np.logical_or(y>ke01_line, x>0.32), y>seyfert_liner_line)
        region_liner = np.logical_and(np.logical_or(y>ke01_line, x>0.32), y<seyfert_liner_line)
        region_starforming = np.logical_and(y<ke01_line, x<0.32)
        if is_masked(x) or is_masked(y):
            return region_seyfert.filled(False), region_liner.filled(False), region_starforming.filled(False)
        else:
            return region_seyfert, region_liner, region_starforming
