import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm, colors
from astropy import cosmology
from astropy.io import fits
import os
import urllib.request as urlib
import collections
import logging
from dl import queryClient as qc
from dl.helpers.utils import convert


#LOGGING SETUP
def setup_logger(name, log_file, formatter, level=logging.INFO):
    '''Sets up a logger
    Inputs:
    - name of logger (string)
    - name of log file (strong)
    - formatter (logging.formatter object)
    - logging level
    
    Outputs:
    - logger object
    '''

    handler = logging.FileHandler(log_file, mode='w')        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

def stringify(arr):
    '''Turns array of various values into a string
    Inputs:
    - arr (array)
    
    Outputs:
    - string
    '''
    output = ''
    for i in arr:
        output = output + str(i) + ' '
    return output

#FUNCTIONS FOR REDSHIFT PREDICTOR
def process_zvals(filename):
    '''Function to turn a txt table of zvals into a pandas frame
    Inputs: 
    - filename (txt file)

    Outputs:
    - pandas frame
    
    '''
    zvals = pd.DataFrame(np.loadtxt(filename, dtype=str, skiprows=0))
    zvals.columns = zvals.iloc[0]
    zvals = zvals.drop(0)
    zvals = zvals.astype(float)
    zvals = zvals.reset_index(drop=True)

    return zvals

def compute_slope(zvals, coltype, mrange=.5):
    '''Function to compute a redshift linelist of red sequence galaxies
    Inputs:
    - zvals (pandas frame of zvals)
    - color type ('gr' or 'rz')
    - mrange (range of magnitudes for each line)

    Outputs:
    - list of lines in format [(minmag, maxmag), (mincol, maxcol), redshift]
    '''
    #coltype either 'gr' or 'rz'
    if coltype == 'gr':
        mag = 'r_mag'
        col = 'g-r_col'
        slope = 'g-r_slope'
    elif coltype == 'rz':
        mag = 'z_mag'
        col = 'r-z_col'
        slope = 'r-z_slope'
    else:
        raise ValueError("Unknown color type.")
    
    linelist = []
    for i in range(len(zvals)):
        if zvals[mag][i] != -10:
            minmag = zvals[mag][i] + mrange
            maxmag = zvals[mag][i] - mrange

            mincol = zvals[col][i] + zvals[slope][i]*mrange
            maxcol = zvals[col][i] - zvals[slope][i]*mrange
            
            line = [(minmag, maxmag), (mincol, maxcol), zvals['Redshift'][i]]
            linelist.append(line)
    
    return linelist

def values_from_index_list(value_list, indicies, i=0):
    '''
    value_list such as predictor.gr_perp or predictor.gr_mags
    indicies in flattened array
    i: which column to look at in the case of gr_perp
    '''
    if len(np.shape(value_list)) == 2:
        vals = np.array(value_list)[:,i]
    elif len(np.shape(value_list)) == 1:
        vals = np.array(value_list)
    val_list = np.broadcast_to(vals, (len(indicies),len(vals)))
    return val_list[np.arange(len(val_list)),indicies]
    
def find_intersection(x1,y1,x2,y2,x3,y3,x4,y4):
    px = ( (x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4) ) / ( (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) ) 
    py = ( (x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4) ) / ( (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) )
    return [px, py]

def get_alt_point(x, y, m):
    return (x+1, m+y)

def flux_to_mag(flux):
    return 22.5 - (2.5*np.log10(flux))

def single_slope(zval, mrange, coltype):
    if coltype=='gr':
        minmag = zval['r_mag'] + mrange
        maxmag = zval['r_mag'] - mrange
        mincol = zval['g-r_col'] + zval['g-r_slope']*mrange
        maxcol = zval['g-r_col'] - zval['g-r_slope']*mrange
    elif coltype=='rz':
        minmag = zval['z_mag'] + mrange
        maxmag = zval['z_mag'] - mrange
        mincol = zval['r-z_col'] + zval['r-z_slope']*mrange
        maxcol = zval['r-z_col'] - zval['r-z_slope']*mrange
    else:
        raise ValueError("Unknown color type.")

    return [(minmag, maxmag), (mincol, maxcol)]

def plot_plain(gr_slopes, rz_slopes, gr_red, rz_red):

    fig = plt.figure(figsize=(10,8))
    ax1 = fig.add_subplot()
    ax2 = ax1.twinx()

    NUM_grCOLORS = len(gr_slopes)
    NUM_rzCOLORS = len(rz_slopes)
    gr_cm = plt.get_cmap('Blues')
    rz_cm = plt.get_cmap('Reds')
    ax1.set_prop_cycle(color=[gr_cm(1.*i/NUM_grCOLORS) for i in range(NUM_grCOLORS)])
    ax2.set_prop_cycle(color=[rz_cm(1.*i/NUM_rzCOLORS) for i in range(NUM_rzCOLORS)])

    ax1.set_xlabel('Magnitude'); ax1.set_ylabel('g-r'); ax2.set_ylabel('r-z')

    for i in range(len(gr_slopes)):
        ax1.plot(gr_slopes[i][0],gr_slopes[i][1])
    for i in range(len(rz_slopes)):
        ax2.plot(rz_slopes[i][0],rz_slopes[i][1])

    axins1 = ax1.inset_axes((0.05,1-0.1,.2,.05))
    gr_norm = colors.Normalize(vmin=np.min(gr_red), vmax=np.max(gr_red))
    cbar = fig.colorbar(mappable=cm.ScalarMappable(norm=gr_norm, cmap='Blues'), cax=axins1, orientation='horizontal',
                    ticks=[np.min(gr_red),np.mean(gr_red),np.max(gr_red)])
    cbar.ax.set_ylabel('g-r')

    axins2 = ax2.inset_axes((0.05,1-0.22,.2,.05))
    rz_norm = colors.Normalize(vmin=np.min(rz_red), vmax=np.max(rz_red))
    cbar = fig.colorbar(mappable=cm.ScalarMappable(norm=rz_norm, cmap='Reds'), cax=axins2, orientation='horizontal',
                    ticks=[np.min(rz_red),np.mean(rz_red),np.max(rz_red)])
    cbar.ax.set_ylabel('r-z')

    plt.show()

def plot_pred(gr_slopes, rz_slopes, gr_red, rz_red, gr_prediction, rz_prediction,
        gr, r, rz, z, zvals, gr_mask, rz_mask, mrange=0.5):

    fig = plt.figure(figsize=(10,8))
    ax1 = fig.add_subplot()
    ax2 = ax1.twinx()

    NUM_grCOLORS = len(gr_slopes)
    NUM_rzCOLORS = len(rz_slopes)
    gr_cm = plt.get_cmap('Blues')
    rz_cm = plt.get_cmap('Reds')
    ax1.set_prop_cycle(color=[gr_cm(1.*i/NUM_grCOLORS) for i in range(NUM_grCOLORS)])
    ax2.set_prop_cycle(color=[rz_cm(1.*i/NUM_rzCOLORS) for i in range(NUM_rzCOLORS)])

    ax1.set_xlabel('Magnitude'); ax1.set_ylabel('g-r'); ax2.set_ylabel('r-z')

    for i in range(len(gr_slopes)):
        ax1.plot(gr_slopes[i][0],gr_slopes[i][1])
    for i in range(len(rz_slopes)):
        ax2.plot(rz_slopes[i][0],rz_slopes[i][1])

    axins1 = ax1.inset_axes((0.05,1-0.1,.2,.05))
    gr_norm = colors.Normalize(vmin=np.min(gr_red), vmax=np.max(gr_red))
    cbar = fig.colorbar(mappable=cm.ScalarMappable(norm=gr_norm, cmap='Blues'), cax=axins1, orientation='horizontal',
                    ticks=[np.min(gr_red),np.mean(gr_red),np.max(gr_red)])
    cbar.ax.set_ylabel('g-r')

    axins2 = ax2.inset_axes((0.05,1-0.22,.2,.05))
    rz_norm = colors.Normalize(vmin=np.min(rz_red), vmax=np.max(rz_red))
    cbar = fig.colorbar(mappable=cm.ScalarMappable(norm=rz_norm, cmap='Reds'), cax=axins2, orientation='horizontal',
                    ticks=[np.min(rz_red),np.mean(rz_red),np.max(rz_red)])
    cbar.ax.set_ylabel('r-z')

    #--
    gr_idx = gr_prediction[1]
    rz_idx = rz_prediction[1]

    gr_slope = single_slope(zvals[gr_mask].iloc[[gr_idx]], mrange, coltype='gr')
    rz_slope = single_slope(zvals[rz_mask].iloc[[rz_idx]], mrange, coltype='rz')
    #--

    #gr
    ax1.plot(gr_slope[0], gr_slope[1], c='dodgerblue', ls=':', alpha=.7)
    ax1.scatter(r, gr, c='dodgerblue', marker='*', s=20, zorder=5)
    
    #rz
    ax2.plot(rz_slope[0], rz_slope[1], c='orangered', ls=':', alpha=.7)
    ax2.scatter(z, rz, c='orangered', marker='*', s=20, zorder=5)

    plt.show()

#------------------

#FUNCTIONS FOR CLUSTER FINDER

def catalog_distance(RA, DEC, data):
    return np.linalg.norm(np.subtract([RA*np.ones_like(data['ra']), DEC*np.ones_like(data['dec'])],
                           [data['ra'], data['dec']]) * [np.cos(np.radians(data['dec'])), np.ones_like(data['dec'])], axis=0)

def compute_gr(data, idx):
    g = flux_to_mag(data['flux_g'][idx])
    r = flux_to_mag(data['flux_r'][idx])
    return g-r

def compute_rz(data, idx):
    r = flux_to_mag(data['flux_r'][idx])
    z = flux_to_mag(data['flux_z'][idx])
    return r-z

def identify_BCG(RA, DEC, data, Predictor, radius=250, ctol=0.2):
    '''
    Inputs:
    - RA, DEC: coords of initial guess
    - data: the catalog
    - predictor: pre-loaded Predictor object
    - radius: kpc radius in which to perform the search
    
    '''
    #Identify initial BCG guess
    trial_idx = []
    atol=1e-6; rtol=1e-6

    while len(trial_idx) == 0:
        ra_idx = np.argwhere(np.isclose(data['ra'], RA*np.ones_like(data['ra']), atol=atol, rtol=rtol))
        dec_idx = np.argwhere(np.isclose(data['dec'], DEC*np.ones_like(data['dec']), atol=atol, rtol=rtol))
        mask = np.isin(ra_idx, dec_idx)
        trial_idx = ra_idx[mask]
        atol *= 10
        rtol *= 10
    
    distances = []
    for i in trial_idx:
        distances.append(np.linalg.norm(np.subtract([RA, DEC],[data['ra'][i], data['dec'][i]])))
        
    BCG_idx = trial_idx[np.argmin(distances)]
    
    #Check additional candidates
    cosmo = cosmology.LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)

    Predictor.import_test_data(data['flux_g'][BCG_idx],
                        data['flux_r'][BCG_idx],
                        data['flux_z'][BCG_idx],)
    zred = np.mean(Predictor.predict(False),axis=0)[0]
    gr = Predictor.gr_data; rz = Predictor.rz_data

    search_radius = radius * cosmo.arcsec_per_kpc_comoving(zred).value / 3600
    candidates = [catalog_distance(data['ra'][BCG_idx], data['dec'][BCG_idx], data) < search_radius]
    candidate_idx = np.array(list(zip(*np.argwhere(candidates)))[1])

    test = sorted(list(zip(data['flux_r'][candidate_idx], candidate_idx)))[::-1]
    for i in test:
        if np.abs(compute_gr(data, i[1]) - gr) < ctol:
            if np.abs(compute_rz(data, i[1]) - rz) < ctol:
                BCG_idx = i[1]
                return BCG_idx

    return BCG_idx

def plot_BCG(RA, DEC, data, BCG_idx):
    cmp = plt.cm.get_cmap('RdYlBu')
    plt.figure(figsize=(8,6))
    plt.scatter(RA, DEC, s=5000, alpha=.3)
    plt.scatter(data['ra'], data['dec'], s=10, c=data['flux_r'], vmin=0, vmax=50, cmap=cmp)
    plt.scatter(data['ra'][BCG_idx], data['dec'][BCG_idx], s=100, marker='o', alpha=.5)
    plt.colorbar()
    plt.xlim(np.max(data['ra']),np.min(data['ra']))
    plt.ylim(np.min(data['dec']),np.max(data['dec']))
    plt.show()

#------------------

def bin_centers(edges, bins):
    edges = np.linspace(edges[0], edges[1], num=bins+1)
    return 0.5*edges[1:] + 0.5*edges[:-1]

#------------------

def calc_area(radius):
    '''
    Calculates true angular size of a circular patch of the sky
    '''
    a = np.pi * (radius**2)
    
    return a

def combine_redshifts(arg1, arg2=None):
    #By default this function takes a single Pandas dataframe with columns 'gr_redshift' and 'rz redshift'.
    #If you give it two arguments in form [gr_redshift], [rz_redshift], it will just combine those.
    #If you give it two scalar arguments in form (gr, rz), it will return a scalar.

    if arg2 is None: #if only a dataframe is given...
        redshift = pd.Series(np.mean([arg1['gr_redshift'].astype('float64'), arg1['rz_redshift'].astype('float64')], axis=0))
        #redshift = arg1['rz_redshift'].astype('float64')
        redshift[arg1['gr_redshift'] < .35] = arg1['gr_redshift'][arg1['gr_redshift'] < .35]
        redshift[arg1['rz_redshift'] > .4] = arg1['rz_redshift'][arg1['rz_redshift'] > .4]
        redshift[arg1['gr_redshift'] > .6] = arg1['rz_redshift'][arg1['gr_redshift'] > .6]

    else: 
        if isinstance(arg1, (collections.Sequence, np.ndarray, pd.Series)): #two redshift lists are given...
            redshift = pd.Series(np.mean([arg1.astype('float64'), arg2.astype('float64')], axis=0))
            #redshift = arg2
            redshift[arg1 < .35] = arg1[arg1 < .35]
            redshift[arg2 > .4] = arg2[arg2 > .4]
            redshift[arg1 > .6] = arg2[arg1 > .6]
        else:
            if arg1 > .6:
                return arg2
            elif arg2 > .4:
                return arg2
            elif arg1 < .35:
                return arg1

            else:
                return np.nanmean([arg1, arg2])

    return redshift

#------------------

def calc_magnitude_error(flux, flux_p, flux_m):
    return np.abs(np.maximum((flux_to_mag(flux_p) - flux_to_mag(flux)), (flux_to_mag(flux) - flux_to_mag(flux_m))))

def add_quadrature(x1, x2):
    return np.sqrt((x1**2) + (x2**2))

def predict_magnitudes_max_min(clus, candidate_idx=None):
    if candidate_idx is None:
        redshifts_max = clus.Predictor.predict_from_values(
            (clus.catalog['r'] - clus.catalog['r_e']), 
            (clus.catalog['z'] - clus.catalog['z_e']), 
            (clus.catalog['g'] - clus.catalog['r'] - add_quadrature(clus.catalog['g_e'], clus.catalog['r_e'])), 
            (clus.catalog['r'] - clus.catalog['z'] - add_quadrature(clus.catalog['r_e'], clus.catalog['z_e'])), 
            False)
        
        redshifts_min = clus.Predictor.predict_from_values(
            (clus.catalog['r'] + clus.catalog['r_e']), 
            (clus.catalog['z'] + clus.catalog['z_e']), 
            (clus.catalog['g'] + clus.catalog['r'] + add_quadrature(clus.catalog['g_e'], clus.catalog['r_e'])), 
            (clus.catalog['r'] + clus.catalog['z'] + add_quadrature(clus.catalog['r_e'], clus.catalog['z_e'])), 
            False)

    else:
        redshifts_max = clus.Predictor.predict_from_values(
            (clus.catalog['r'] - clus.catalog['r_e'])[candidate_idx], 
            (clus.catalog['z'] - clus.catalog['z_e'])[candidate_idx], 
            (clus.catalog['g'] - clus.catalog['r'] - add_quadrature(clus.catalog['g_e'], clus.catalog['r_e']))[candidate_idx], 
            (clus.catalog['r'] - clus.catalog['z'] - add_quadrature(clus.catalog['r_e'], clus.catalog['z_e']))[candidate_idx], 
            False)
        
        redshifts_min = clus.Predictor.predict_from_values(
            (clus.catalog['r'] + clus.catalog['r_e'])[candidate_idx], 
            (clus.catalog['z'] + clus.catalog['z_e'])[candidate_idx], 
            (clus.catalog['g'] + clus.catalog['r'] + add_quadrature(clus.catalog['g_e'], clus.catalog['r_e']))[candidate_idx], 
            (clus.catalog['r'] + clus.catalog['z'] + add_quadrature(clus.catalog['r_e'], clus.catalog['z_e']))[candidate_idx], 
            False)
    
    return redshifts_max, redshifts_min

#------------

def physical_area(dec, size, ra_min, ra_max, dec_min, dec_max):
    a = ((size*np.abs(np.cos(np.radians(dec_max))) + size*np.abs(np.cos(np.radians(dec_min)))) / 2) * size 
    return a

outputs = os.getcwd()
url = 'https://www.legacysurvey.org/viewer/ls-dr9/cat.fits?ralo={}&rahi={}&declo={}&dechi={}'
#url = 'https://www.legacysurvey.org/viewer/dr8/cat.fits?ralo={}&rahi={}&declo={}&dechi={}'

def download_cat(ra, dec, size=0.03, skip_masking=False, verbose=True):
    ra_min = ra-(size/2)
    ra_max = ra+(size/2)
    dec_min = dec-(size/2)
    dec_max = dec+(size/2)

    if not skip_masking:
        ca_result = physical_area(dec, size, ra_min, ra_max, dec_min, dec_max)

        if not ca_result:
            raise ValueError('Skipped due to masking.')

    outname = '/'+str(ra)+'_'+str(dec)+'.fits'
    if not os.path.isfile(outputs+outname):
        try:
            urlib.urlretrieve(url.format(
                ra_min, ra_max, dec_min, dec_max), outputs+outname)
        except Exception as e:
            logging.warning('Catalog at [%s , %s] did not download.', ra, dec)
            logging.warning('%s', e)
            if verbose:
                print('\x1b[31m Catalog at [', ra, dec, '] timed out. :( \x1b[0m')
                print(e)
            return None
        logging.info('Catalog at [%s , %s] has been downloaded.', ra, dec)
        if verbose:
            print('\x1b[32m Catalog at [', ra, dec,'] has been downloaded. \x1b[0m')
    else:
        logging.info('Catalog at [%s , %s] was already downloaded.', ra, dec)
        if verbose:
            print('\x1b[33m Catalog at [', ra, dec,'] was already downloaded. \x1b[0m')

    if not skip_masking:
        return ca_result
    return None

types = ['DEV', 'EXP', 'REX', 'SER']
def load_cat(ra, dec, i, bitmasks=True, verbose=True, remove=True):
    filename = str(ra)+'_'+str(dec)+'.fits'
    try:
        with fits.open(filename) as hdul:
            dat = hdul[1].data
        if remove:
            os.remove(filename)
        if bitmasks:
            return dat[np.isin(dat['type'],types)]
        else:
            return dat
    except Exception as e:
        logging.warning('%s', e)
        if verbose:
            print('Exception for catalog', i)
            print(e)
            
query = '''SELECT {} FROM {}.tractor WHERE ra>{} AND ra<{} AND dec>{} AND dec<{} AND type IN ('DEV', 'EXP', 'REX', 'SER')'''
#query = '''SELECT {} FROM ls_dr8.tractor WHERE ra>{} AND ra<{} AND dec>{} AND dec<{} AND type IN ('DEV', 'EXP', 'REX', 'SER')'''
default_cols = '''ra, dec, flux_g, flux_r, flux_z, flux_ivar_g, flux_ivar_r, flux_ivar_z, mw_transmission_g, mw_transmission_r, mw_transmission_z, maskbits, ls_id, sersic, shape_r, shape_e1, shape_e2'''

def sql_cat(ra, dec, size=0.03, columns=default_cols, catalog='ls_dr9', verbose=False):
    """Load data directly into a pandas dataframe from the SQL database.

    Args:
        ra (float): Center coordinate of cutout
        dec (float): Center coordinate of cutout
        size (float, optional): Diameter (width) of the cutout. Defaults to 0.03.
        verbose (bool, optional): Verbose? Defaults to False.
        
    Returns:
        Pandas DataFrame with catalog info
    """    
    
    #Explicitly calculate the bounds of the cutout.
    ra_min = ra-(size/2)
    ra_max = ra+(size/2)
    dec_min = dec-(size/2)
    dec_max = dec+(size/2)
    
    #Try to download the data.
    try:
        response = qc.query(adql = query.format(columns, catalog, ra_min, ra_max, dec_min, dec_max), fmt='csv')
        df = convert(response)
        logging.info('Catalog at [%s , %s] has been direct loaded from sql.', ra, dec)
        if verbose:
            print('\x1b[32m Catalog at [', ra, dec,'] has been direct loaded from sql. \x1b[0m')
        return df
    except Exception as e:
            logging.warning('Catalog at [%s , %s] failed to direct load from sql. :(', ra, dec)
            logging.warning('%s', e)
            if verbose:
                print('\x1b[31m Catalog at [', ra, dec, '] failed to direct load from sql. :( \x1b[0m')
                print(e)
