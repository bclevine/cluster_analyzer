import numpy as np
import pandas as pd
from helpers import *
from redshift_predictor import *
from cluster import Cluster
from astropy.io import fits
import time

import warnings
warnings.filterwarnings("ignore")

#Load predictor model
pred = Predictor('RSmodel_July2020.txt')
pred.compute_arrays()

#Load catalogs
filename = 'DECALS Cutout Grabber/catalogs.nosync/{}_{}.fits'
cutout_list = np.loadtxt('DECALS Cutout Grabber/catalogs.txt')
catalog_list = []
indicies = []
for i, j in enumerate(cutout_list[:]):
    try:
        hdul = fits.open(filename.format(np.format_float_positional(j[0]),
                                   np.format_float_positional(j[1])))
        catalog_list.append(hdul[1].data)
        hdul.close()
        indicies.append(i)
    except Exception:
        print('Exception for catalog', i)
        catalog_list.append(None)

#Run analysis
start = time.time() #start time
statistics = []
for i in range(len(indicies)):
    
    try:
        stage = 'Load'
        c = cutout_list[indicies[i]]
        RA = c[0]; DEC = c[1]
        clus = Cluster(catalog_list[i], RA, DEC, pred)
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
        #print(output)
    except Exception:
        print("Failed to process catalog", i)
        print('Failure stage:', stage)
    
end = time.time()
print("Elapsed time is  {}".format(end-start))

#Save statistics
print('Length of statistics:', len(statistics))
with open('all_stats.npy', 'wb') as f:
    np.save(f, statistics)