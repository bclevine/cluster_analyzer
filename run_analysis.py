import numpy as np
import pandas as pd
from helpers import *
from redshift_predictor import *
from cluster import Cluster
from astropy.io import fits
import time
import urllib.request as urlib
import os

import warnings
warnings.filterwarnings("ignore")

#Load predictor model
pred = Predictor('RSmodel_July2020.txt')
pred.compute_arrays()

#Load catalogs
filename = 'DECALS Cutout Grabber/catalogs.nosync/{}_{}.fits'
cutout_list = np.loadtxt('DECALS Cutout Grabber/catalogs.txt')
url = 'https://www.legacysurvey.org/viewer/ls-dr9/cat.fits?ralo={}&rahi={}&declo={}&dechi={}'

directory = os.getcwd()
outputs = directory+'/DECALS Cutout Grabber/catalogs.nosync'

#Run analysis
start = time.time() #start time
statistics = []
for i in range(len(cutout_list)):
    try:
        #-----
        stage = 'Download Small Catalog'
        ra, dec, size = cutout_list[i]
        ra_min = ra-size; ra_max = ra+size
        dec_min = dec-size; dec_max = dec+size
        outname = '/'+str(ra)+'_'+str(dec)+'.fits'
        
        urlib.urlretrieve(url.format(ra_min, ra_max, dec_min, dec_max), outputs+outname)
        print('Catalog at [', ra, dec, '] has been downloaded.')

        #-----
        stage = 'Load'
        with fits.open(filename.format(np.format_float_positional(j[0]), np.format_float_positional(j[1]))) as hdul:
            dat = hdul[1].data
        clus = Cluster(dat, ra, dec, pred)

        #-----
        stage = 'Identify'
        clus.identify_BCG(iterate=True)
        stage = 'Cluster z'
        clus.calc_redshifts(radius=500)
        stage = 'Download'
        clus.download_surroundings()
        
        stage = 'Large z'
        clus.calc_redshifts(large=True)

        stage = 'Richness'
        clus.calc_richness()

        stage = 'Return Stats'
        output = clus.return_statistics()
        statistics.append(output)

    except Exception:
        print("Failed to process catalog", i)
        print('Failure stage:', stage)
    
end = time.time()
print("Elapsed time is  {}".format(end-start))

#Save statistics
print('Length of statistics:', len(statistics))
with open('nonlensing_stats.npy', 'wb') as f:
    np.save(f, statistics)