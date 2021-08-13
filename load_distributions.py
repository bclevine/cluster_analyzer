import numpy as np
from helpers import *
from redshift_predictor import *
from cluster import Cluster
from astropy.io import fits
import os
import urllib.request as urlib

from multiprocessing import Pool
import warnings
warnings.filterwarnings("ignore")

#-------

pred = Predictor('RSmodel_July2020.txt')
pred.compute_arrays()
cutout_list = np.loadtxt('DECALS Cutout Grabber/locations.txt')
url = 'https://www.legacysurvey.org/viewer/ls-dr9/cat.fits?ralo={}&rahi={}&declo={}&dechi={}'
outputs=os.getcwd()

def load_cat(ra, dec, i, verbose=True):
    filename = str(ra)+'_'+str(dec)+'.fits'
    try:
        with fits.open(filename) as hdul:
            dat = hdul[1].data
        os.remove(filename)
        return dat
    except Exception as e:
        if verbose:
            print('Exception for catalog', i)
            print(e)

def download_cat(ra, dec, size=0.03):
    ra_min = ra-size; ra_max = ra+size
    dec_min = dec-size; dec_max = dec+size

    outname = '/'+str(ra)+'_'+str(dec)+'.fits'
    if not os.path.isfile(outputs+outname):
        try:
            urlib.urlretrieve(url.format(ra_min, ra_max, dec_min, dec_max), outputs+outname)
        except Exception as e:
            print('\x1b[31m Catalog at [', ra, dec, '] timed out. :( \x1b[0m')
            print(e)
            return None

        print('\x1b[32m Catalog at [', ra, dec, '] has been downloaded. \x1b[0m')
    else:
        print('\x1b[33m Catalog at [', ra, dec, '] was already downloaded. \x1b[0m')

def calc_background(i):
    try:
        c = cutout_list[i]
        RA = c[0]; DEC = c[1]
    
        stage = 'Download'
        download_cat(RA, DEC)
        
        stage = 'Load'
        clus = Cluster(load_cat(RA, DEC, i), RA, DEC, pred)
        #-----
        stage = 'BCG'
        clus.identify_BCG(iterate=True)
        
        stage = 'Redshifts'
        clus.calc_redshifts(radius=500)
        
        stage = 'Calc Background'
        return clus.calc_background()
    except Exception:
        print("Failed to process catalog", i)
        print("Failure stage:", stage)

def calc_distributions():
    with Pool() as p:
        background_distributions = p.map(calc_background, list(range(len(cutout_list))))
    return background_distributions

if __name__ == '__main__': 
    a = np.array(calc_distributions(), dtype=object)
    with open('mapdata.npy', 'wb') as f:
        np.save(f, a)