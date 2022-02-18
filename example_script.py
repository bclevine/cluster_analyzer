#CONFIG: CHANGE THESE AS APPLICABLE
#----------------------------------
logfile = 'example_richnesses.log' # <-- The name of the log file
savefile = 'example_rich.npy' # <-- The name of the final output 
input_catalog = 'COOLLAMPS_decals_springlowz_searchpositions.txt' # <-- The name of the input catalog (list of [ra, dec] coordinates)
deg_radius = .05 # <-- Cutout radius in degrees
#----------------------------------
#END OF CONFIG

#IMPORTS
#Logging
import logging
logging.basicConfig(filename=logfile, filemode='w', level=logging.INFO)
    
#Computation & Algorithms
import numpy as np
from helpers import *
from redshift_predictor import *
from cluster import Cluster
from astropy.io import fits
import os
import time
from wrapt_timeout_decorator import *
import argparse
from helpers import download_cat, load_cat

#Multithreading
from multiprocessing import Pool
import warnings
warnings.filterwarnings("ignore")


#PARSE ARGUMENTS
def argument_parser():
    """Function that parses the arguments passed when running the script

    Returns:
        ArgumentParser object: the arguments
    """    
    '''
    Function that parses the arguments passed while running a script
    '''
    result = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    result.add_argument('-n', dest='n_threads', type=int, default=25) 
    result.add_argument('-m', dest='MASKING_CORRECTION', type=bool, default=True) 
    result.add_argument('-l', dest='length', type=int, default=-10)
    result.add_argument('-t', dest='timeout', type=int, default=60)
    result.add_argument('-v', dest='verbose', type=bool, default=False)
    result.add_argument('-s', dest='sigma', type=float, default=1)

    return result

args = argument_parser().parse_args()

#HANDLE MASKING, IF APPLICABLE
MASKING_CORRECTION = args.MASKING_CORRECTION

#IMPORT TQDM PROGRESS BAR
if not args.verbose:
    from tqdm.contrib.concurrent import process_map
    
#INITIALIZE PREDICTOR AND COSMOLOGY
pred = Predictor('RSmodel_Nov2021.txt')
pred.compute_arrays()
cosmo = cosmology.WMAP9    

#LOAD IN THE SEARCH POSITION CATALOG
cat = pd.DataFrame(np.loadtxt(input_catalog), columns=['ra','dec'])

#Note — if you're using a csv (ie. input catalog is separated by commas not spaces, use the commented line below)
#cat = pd.read_csv(input_catalog, names=['ra','dec'])

cat = cat.astype({'ra':'float', 'dec':'float',})

#IF WE ARE CORRECTING FOR SURVEY GEOMETRY...
if MASKING_CORRECTION:
    from glob import iglob
    from bisect import bisect_left

    #IMPORT RANDOMS
    randoms = []
    for i in iglob('../../../data/blevine/randoms/rand_sorted_*.fits'):
        randoms.append(fits.open(i)[1].data)

    #COUNT RANDOMS FUNCTION
    def count_randoms(ramin, ramax, decmin, decmax):
        n = 0
        for i in randoms:
            left_index = bisect_left(i['RA'], ramin)
            right_index = bisect_left(i['RA'], ramax) - 1
            n += np.count_nonzero(((i['DEC'][left_index:right_index]>decmin) & (i['DEC'][left_index:right_index]<decmax)))
        return n

    #CORRECTION FUNCTION
    def calc_correction(size, ra, dec):
        ra_min = ra-(size/2)
        ra_max = ra+(size/2)
        dec_min = dec-(size/2)
        dec_max = dec+(size/2)

        a = ((size*np.abs(np.cos(np.radians(dec_max))) + size*np.abs(np.cos(np.radians(dec_min)))) / 2) * size 
        #1 sq arcmin = 2.78e-4 sq deg ~~ 20.5 randoms
        
        n_randoms = count_randoms(ra_min, ra_max, dec_min, dec_max)
        expected = (20.5/2.78e-4)*a

        return n_randoms/expected
    
#MAKE LIST OF INDICIES
inds = list(range(len(cat)))
np.random.shuffle(inds)

if args.length != -10:
    inds = inds[:args.length]
    
print('BEGINNING RUN WITH', args.n_threads, 'THREADS.')
print('LENGTH OF INPUT CATALOG:', len(inds))
print('MAX TIMEOUT:', args.timeout)
print('VERBOSE:', args.verbose)
print('----------------------------------------------')

#CALCULATE RICHNESS STATISTICS
@timeout(args.timeout, use_signals=False)
def calc_richnesses(j):
    stage = 'None'
    start = time.time()
    try:
        i = inds[j]
        RA = cat['ra'][i]
        DEC = cat['dec'][i]

        stage = 'Download'
        download_cat(RA, DEC, size=deg_radius*2, skip_masking=True, verbose=args.verbose) #this is the total size not the radius

        stage = 'Load'
        clus = Cluster(load_cat(RA, DEC, i, verbose=args.verbose), RA, DEC, pred)
        # -----
        stage = 'Identify BCG'
        clus.identify_BCG(use_given_coords=False)

        stage = 'Redshifts'
        clus.calc_redshifts(full_cat=True)

        stage = 'Calc Richness'
        clus.calc_richness(comparison_redshift='z_BCG', comparison_tolerance=.1, use_uncertainty=True, sigma=args.sigma)

        if MASKING_CORRECTION:
            stage = 'Calculate Masking'
            mask_result = calc_correction(deg_radius*2, RA, DEC)
            
        logging.info('%s complete.', i)
        if args.verbose:
            print(j, 'complete.')
            
        if os.path.exists(str(RA)+'_'+str(DEC)+'.fits'):
            os.remove(str(RA)+'_'+str(DEC)+'.fits')
        
        #RETURNED VALUES
        #There are a whole lot of things you can return here. Just make an array of whatever you want.
        #The main values are 
        # - clus.richness (richness)
        # - clus.z_BCG (BCG redshift)
        # - clus.mean_z (mean redshift of all cluster members)
        # - i (index of cluster; use this to match back to the initial coordinate list)
        # - clus.catalog['XXX'][clus.BCG_idx] (any arbitrary BCG parameters such as position, magnitude, etc)
        # - mask_result (the mask correction fraction)
        #You can use anything else from the Cluster object. Contact me if there's something else you need returned.
        
        if MASKING_CORRECTION:
            return [clus.richness, clus.z_BCG, clus.mean_z, i, mask_result,
                    clus.catalog['ra'][clus.BCG_idx], clus.catalog['dec'][clus.BCG_idx], clus.catalog['r'][clus.BCG_idx], clus.catalog['z'][clus.BCG_idx], clus.initial_z_BCG]
        return [clus.richness, clus.z_BCG, clus.mean_z, i, 
                clus.catalog['ra'][clus.BCG_idx], clus.catalog['dec'][clus.BCG_idx], clus.catalog['r'][clus.BCG_idx], clus.catalog['z'][clus.BCG_idx], clus.initial_z_BCG]
        
    except Exception as e:
        logging.warning('Failed to process catalog %s', i)
        logging.warning('Failure stage: %s', stage)
        logging.warning('%s', e)
        if args.verbose:
            print("Failed to process catalog", i)
            print("Failure stage:", stage)
            print(e)

        try:
            i = inds[j]

            RA = cat['ra'][i]
            DEC = cat['dec'][i]

            if os.path.exists(str(RA)+'_'+str(DEC)+'.fits'):
                os.remove(str(RA)+'_'+str(DEC)+'.fits')
        
        except:
            return None

#HANDLE TIMEOUTS
def timeout_func(i):
    try:
        return calc_richnesses(i)
    except Exception:
        try:
            idx = inds[i]

            RA = cat['ra'][idx]
            DEC = cat['dec'][idx]

            if os.path.exists(str(RA)+'_'+str(DEC)+'.fits'):
                os.remove(str(RA)+'_'+str(DEC)+'.fits')
                
            logging.warning('%s timed out ([%s , %s]).', i, RA, DEC)
            if args.verbose:
                print(i, 'timed out.')
                
        except:
            logging.warning('%s timed out (additional issue with loading data also occured).', i)
            return None

        if args.verbose:
            print(i, 'timed out.')

#WRAPPER FUNCTION FOR MULTITHREADING
def do_calc():
    if args.verbose:
        with Pool(args.n_threads) as p:
            output = p.map(
                timeout_func, range(len(inds)))
        return output
    else:
        output = process_map(timeout_func, range(len(inds)), max_workers=args.n_threads)
        return output

#MAIN
if __name__ == '__main__':
    logging.info('Started')
    start = time.time()

    a = np.array(do_calc(), dtype=object)
    print('Saving...')
    with open(savefile, 'wb') as f:
        np.save(f, np.array([a]))

    end = time.time()
    print("Elapsed time is {}".format(end-start))
    logging.info('Finished')