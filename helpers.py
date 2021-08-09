import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import cm, colors
from astropy import cosmology

#------------------

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

#------------------

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

#------------------

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

#------------------

def flux_to_mag(flux):
    return 22.5 - (2.5*np.log10(flux))

#------------------

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

def catalog_distance(RA, DEC, data):
    return np.linalg.norm(np.subtract([RA*np.ones_like(data['ra']), DEC*np.ones_like(data['dec'])],
                           [data['ra'], data['dec']]),axis=0)

#------------------

def compute_gr(data, idx):
    g = flux_to_mag(data['flux_g'][idx])
    r = flux_to_mag(data['flux_r'][idx])
    return g-r

def compute_rz(data, idx):
    r = flux_to_mag(data['flux_r'][idx])
    z = flux_to_mag(data['flux_z'][idx])
    return r-z

#------------------

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

#------------------

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