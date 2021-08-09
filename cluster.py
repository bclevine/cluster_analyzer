import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from astropy import cosmology
import matplotlib.image as mpimg
import os
import urllib.request as urlib
import socket
from astropy.io import fits
from multiprocessing import Pool
#--
from helpers import *
#from redshift_predictor import Predictor
#--

class Cluster:
    def __init__(self, data, RA, DEC, Predictor, radius=500, ctol=0.2):
        '''
        data: catalog cutout for the cluster
        RA, DEC: initial guess RA and DEC
        Predictor: pre-loaded Predictor object
        --Optional:--
        radius: search radius of cluster in kpc
        ctol: color separation tolerance
        '''
        self.catalog = pd.DataFrame({'ra':data['ra'].byteswap().newbyteorder(), 
            'dec':data['dec'].byteswap().newbyteorder(),
            'g':flux_to_mag(data['flux_g'].byteswap().newbyteorder()), 
            'r':flux_to_mag(data['flux_r'].byteswap().newbyteorder()), 
            'z':flux_to_mag(data['flux_z'].byteswap().newbyteorder()),
            })
        self.catalog['gr'] = self.catalog['g'] - self.catalog['r']
        self.catalog['rz'] = self.catalog['r'] - self.catalog['z']
        self.Predictor = Predictor
        self.radius = radius
        self.ctol = ctol
        self.RA = RA
        self.DEC = DEC

    def identify_BCG(self, radius=None, ctol=None, iterate=True):
        if radius==None:
            radius = self.radius
        if ctol==None:
            ctol = self.ctol

        #Identify initial BCG guess
        trial_idx = []
        atol=1e-6; rtol=1e-6

        while len(trial_idx) == 0:
            ra_idx = np.argwhere(np.isclose(self.catalog['ra'], self.RA*np.ones_like(self.catalog['ra']), 
                atol=atol, rtol=rtol))
            dec_idx = np.argwhere(np.isclose(self.catalog['dec'], self.DEC*np.ones_like(self.catalog['dec']), 
                atol=atol, rtol=rtol))
            mask = np.isin(ra_idx, dec_idx)
            trial_idx = ra_idx[mask]
            atol *= 10
            rtol *= 10
        
        distances = []
        for i in trial_idx:
            distances.append(np.linalg.norm(np.subtract([self.RA, self.DEC],
                [self.catalog['ra'][i], self.catalog['dec'][i]])))
            
        BCG_idx = trial_idx[np.argmin(distances)]
        self.BCG_idx = BCG_idx

        #Check additional candidates
        cosmo = cosmology.LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)

        zred = np.mean(self.Predictor.predict_from_values(self.catalog['r'][BCG_idx], self.catalog['z'][BCG_idx],
            self.catalog['gr'][BCG_idx], self.catalog['rz'][BCG_idx], False), axis=0)[0]

        search_radius = radius * cosmo.arcsec_per_kpc_comoving(zred).value / 3600

        found = False
        iterations = 0

        while found==False:
            candidates = [catalog_distance(self.catalog['ra'][BCG_idx], 
                self.catalog['dec'][BCG_idx], self.catalog) < search_radius]
            candidate_idx = np.array(list(zip(*np.argwhere(candidates)))[1])

            for i in sorted(list(zip(self.catalog['r'][candidate_idx], candidate_idx))):
                #Check whether colors are within color tolerances
                if np.abs(self.catalog['gr'][i[1]] - self.catalog['gr'][BCG_idx]) < ctol:
                    if np.abs(self.catalog['rz'][i[1]] - self.catalog['rz'][BCG_idx]) < ctol:
                        BCG_idx = i[1]
                        iterations += 1
                        if iterate==False:
                            found = True
                        if BCG_idx == self.BCG_idx:
                            found = True
                        self.BCG_idx = BCG_idx

                        if iterations >= 100:
                            found = True

                        break
        
        self.BCG_idx = BCG_idx
        self.z_BCG = np.mean(self.Predictor.predict_from_values(self.catalog['r'][self.BCG_idx], 
            self.catalog['z'][self.BCG_idx], self.catalog['gr'][self.BCG_idx], 
            self.catalog['rz'][self.BCG_idx], False), axis=0)[0]
        return self.BCG_idx

    def plot_BCG(self, catalog_width=0.03, img=None, pxscale=0.262, imgwidth=250, xscale=1, yscale=1):
        cmp = plt.cm.get_cmap('RdYlBu')
        if img==None:
            plt.figure(figsize=(8,6))
            plt.scatter(self.RA, self.DEC, s=5000, alpha=.3)
            plt.scatter(self.catalog['ra'], self.catalog['dec'], s=10, c=self.catalog['r'], 
                vmax=np.median(self.catalog['r']), vmin=self.catalog['r'][self.BCG_idx], cmap=cmp)
            plt.scatter(self.catalog['ra'][self.BCG_idx], self.catalog['dec'][self.BCG_idx], s=100, marker='o', alpha=.5)
            plt.colorbar()
            plt.xlim(self.RA+catalog_width, self.RA-catalog_width)
            plt.ylim(self.DEC-catalog_width, self.DEC+catalog_width)
            plt.show()
        else:
            catrange = (pxscale*imgwidth/2)/3600
            ramask = [(self.catalog['ra'] < self.RA+catrange) and (self.catalog['ra'] > self.RA-catrange)]
            decmask = [(self.catalog['dec'] < self.DEC+catrange) and (self.catalog['dec'] > self.DEC-catrange)]
            catmask = np.logical_and(ramask, decmask).flatten()
            ra = self.catalog['ra']; dec = self.catalog['dec']

            #-- Renormalize
            ra_px = np.abs(((ra-self.RA)*xscale-catrange)*250/(2*catrange))
            dec_px = np.abs(((dec-self.DEC)*yscale-catrange)*250/(2*catrange))
            #--
            plt.figure(figsize=(8,6))
            plt.imshow(mpimg.imread(img), zorder=0)
            plt.scatter(ra_px[catmask], dec_px[catmask], s=10, c=self.catalog['r'][catmask], 
                vmax=np.median(self.catalog['r'].dropna()), vmin=self.catalog['r'][self.BCG_idx], cmap=cmp, zorder=10)
            plt.scatter(ra_px[self.BCG_idx], dec_px[self.BCG_idx], s=150, marker='o', alpha=.4, zorder=11)
            plt.colorbar()
            plt.show()
            
    def identify_cluster(self, algorithm='color', radius=None, ctol=None, ztol=0.2):
        '''
        algorithms: 
        - 'color' (faster, compares colors within ctol of BCG)
        - 'redshift' (slower, compares redshifts within ztol of BCG)
        '''
        if radius==None:
            radius = self.radius
        if ctol==None:
            ctol = self.ctol

        cosmo = cosmology.LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)
        search_radius = radius * cosmo.arcsec_per_kpc_comoving(self.z_BCG).value / 3600

        candidates = [catalog_distance(self.catalog['ra'][self.BCG_idx], 
            self.catalog['dec'][self.BCG_idx], self.catalog) < search_radius]
        candidate_idx = np.array(list(zip(*np.argwhere(candidates)))[1])

        self.cluster_members = []
        if algorithm=='color':
            for i in candidate_idx:
                if np.abs(self.catalog['gr'][i] - self.catalog['gr'][self.BCG_idx]) < ctol:
                    if np.abs(self.catalog['rz'][i] - self.catalog['rz'][self.BCG_idx]) < ctol:
                        self.cluster_members.append(i)

        elif algorithm=='redshift':
            na_mask = self.catalog.notna()
            for i in candidate_idx:
                if na_mask.iloc[i].all():
                    if np.abs(self.z_BCG - np.mean(self.Predictor.predict_from_values(self.catalog['r'][i], 
                        self.catalog['z'][i], self.catalog['gr'][i], self.catalog['rz'][i], False), axis=0)[0]) < ztol:
                        self.cluster_members.append(i)
        
        else:
            raise ValueError('Unknown algorithm (valid options are color and redshift).')

        return self.cluster_members

    def plot_cluster(self, catalog_width=0.03):
        plt.figure(figsize=(8,6))
        plt.scatter(self.catalog['ra'], self.catalog['dec'], s=10, c='red')
        plt.scatter(self.catalog['ra'][self.cluster_members], self.catalog['dec'][self.cluster_members], s=30, c='blue')
        plt.scatter(self.catalog['ra'][self.BCG_idx], self.catalog['dec'][self.BCG_idx], s=50, c='green')
        plt.xlim(self.RA+catalog_width, self.RA-catalog_width)
        plt.ylim(self.DEC-catalog_width, self.DEC+catalog_width)
        plt.show()

    def calc_redshifts(self, radius=None, large=False):
        if large==False:
            if radius==None:
                radius = self.radius
            cosmo = cosmology.LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)
            search_radius = radius * cosmo.arcsec_per_kpc_comoving(self.z_BCG).value / 3600

            candidates = [catalog_distance(self.catalog['ra'][self.BCG_idx], 
                self.catalog['dec'][self.BCG_idx], self.catalog) < search_radius]
            candidate_idx = np.array(list(zip(*np.argwhere(candidates)))[1])

            redshifts = self.Predictor.predict_from_values(self.catalog['r'][candidate_idx], 
                self.catalog['z'][candidate_idx], self.catalog['gr'][candidate_idx], self.catalog['rz'][candidate_idx], False)
                    
            self.catalog['gr_redshift'] = np.NaN
            self.catalog['rz_redshift'] = np.NaN
            self.catalog['gr_redshift'][candidate_idx] = redshifts[0] 
            self.catalog['rz_redshift'][candidate_idx] = redshifts[1] 

        else:
            #Assert that self.large_catalog exists.
            try:
                self.large_catalog
            except NameError:
                print('Error: Import large catalog first.')
                return None

            #Now calculate redshifts
            if radius==None:
                radius = 5000
            cosmo = cosmology.LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)
            search_radius = radius * cosmo.arcsec_per_kpc_comoving(self.z_BCG).value / 3600

            candidates = [catalog_distance(self.catalog['ra'][self.BCG_idx], 
                self.catalog['dec'][self.BCG_idx], self.large_catalog) > search_radius]
            candidate_idx = np.array(list(zip(*np.argwhere(candidates)))[1])

            '''input_list = list(zip(self.large_catalog['r'][candidate_idx],
                self.large_catalog['z'][candidate_idx],
                self.large_catalog['gr'][candidate_idx],
                self.large_catalog['rz'][candidate_idx],
                np.zeros_like(self.large_catalog['r'][candidate_idx],dtype=bool)))

            with Pool() as p:
                redshifts = p.starmap(self.Predictor.predict_from_values, input_list)'''

            redshifts = self.Predictor.predict_from_values(self.large_catalog['r'][candidate_idx], 
                self.large_catalog['z'][candidate_idx], self.large_catalog['gr'][candidate_idx], 
                self.large_catalog['rz'][candidate_idx], False)

            #gr, rz = list(zip(*redshifts))
            self.large_catalog['gr_redshift'] = np.NaN
            self.large_catalog['rz_redshift'] = np.NaN
            self.large_catalog['gr_redshift'][candidate_idx] = redshifts[0] #list(zip(*gr))[0]
            self.large_catalog['rz_redshift'][candidate_idx] = redshifts[1] #list(zip(*rz))[0]

    #---

    def download_surroundings(self, directory='DECALS Cutout Grabber/catalogs.nosync', verbose=True):
        ra = self.catalog['ra'][self.BCG_idx]
        dec = self.catalog['dec'][self.BCG_idx]
        outname = '/large_'+str(ra)+'_'+str(dec)+'.fits'
        path = os.getcwd()+'/'+directory

        #Check if catalog already exists
        if not os.path.isfile(path+outname):
            url = 'https://www.legacysurvey.org/viewer/ls-dr9/cat.fits?ralo={}&rahi={}&declo={}&dechi={}'

            cosmo = cosmology.LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)
            size = 10000 * cosmo.arcsec_per_kpc_comoving(self.z_BCG).value / 3600

            ra_min = ra-size; ra_max = ra+size
            dec_min = dec-size; dec_max = dec+size

            socket.setdefaulttimeout(60)

            try:
                urlib.urlretrieve(url.format(ra_min, ra_max, dec_min, dec_max),path+outname)
            except Exception as e:
                print('Large catalog at [', ra, dec, '] timed out. :(')
                print(e)
                return None
            if verbose==True:
                print('Large catalog at [', ra, dec, '] has been downloaded.')
        else:
            if verbose==True:
                print('Catalog already exists; did not attempt to download.')

        #Load catalog
        hdul = fits.open(directory+outname)
        data = hdul[1].data
        hdul.close()

        self.large_catalog = pd.DataFrame({'ra':data['ra'].byteswap().newbyteorder(), 
            'dec':data['dec'].byteswap().newbyteorder(),
            'g':flux_to_mag(data['flux_g'].byteswap().newbyteorder()), 
            'r':flux_to_mag(data['flux_r'].byteswap().newbyteorder()), 
            'z':flux_to_mag(data['flux_z'].byteswap().newbyteorder()),
            })
        self.large_catalog['gr'] = self.large_catalog['g'] - self.large_catalog['r']
        self.large_catalog['rz'] = self.large_catalog['r'] - self.large_catalog['z']

    def plot_distribution(self, type, bins=50, range=[0,1.5]):
        '''
        Type: either 'gr' or 'rz'.
        '''
        plt.figure(figsize=(8,6))
        if type=='gr':
            plt.title('Proportion of Objects compared to Normal (g-r redshift)')
            cluster_dist = np.histogram(self.catalog['gr_redshift'].dropna(), bins=bins, density=True, range=range)
            normal_dist = np.histogram(self.large_catalog['gr_redshift'].dropna(), bins=bins, density=True, range=range)
        elif type=='rz':
            plt.title('Proportion of Objects compared to Normal (r-z redshift)')
            cluster_dist = np.histogram(self.catalog['rz_redshift'].dropna(), bins=bins, density=True, range=range)
            normal_dist = np.histogram(self.large_catalog['rz_redshift'].dropna(), bins=bins, density=True, range=range)
        
        vals = np.nan_to_num(cluster_dist[0]/normal_dist[0])
        bin_centers = 0.5 * cluster_dist[1][1:] + 0.5 * cluster_dist[1][:-1]
        plt.plot(bin_centers, vals)
        plt.vlines(self.z_BCG, 0, np.max(vals), color='orange', ls=':')
        plt.xlabel('Redshift'); plt.ylabel('Proportion of Objects')

    def calc_richness(self, bins=50, range=[0,1.5], radius=None):
        self.catalog['redshift'] = pd.Series(np.mean([self.catalog['gr_redshift'].astype('float64'),
            self.catalog['rz_redshift'].astype('float64')], axis=0))
        large_redshift = np.mean([self.large_catalog['gr_redshift'].astype('float64'),
            self.large_catalog['rz_redshift'].astype('float64')], axis=0)
        cluster_dist = np.histogram(self.catalog['redshift'].dropna(), bins=bins, density=True, range=range)
        normal_dist = np.histogram(large_redshift[~np.isnan(large_redshift)], bins=bins, density=True, range=range)
        proportion_dist = np.nan_to_num(cluster_dist[0]/normal_dist[0])
        
        BCG_ind = np.digitize(self.z_BCG, cluster_dist[1]) - 1
        min_idx = BCG_ind
        max_idx = BCG_ind

        while proportion_dist[min_idx - 1] > 1:
            min_idx = min_idx - 1
            
        while proportion_dist[max_idx + 1] > 1:
            max_idx = max_idx + 1

        min_z = cluster_dist[1][min_idx]
        max_z = cluster_dist[1][max_idx+1]

        zmask = np.logical_and((self.catalog['redshift'] > min_z), (self.catalog['redshift'] < max_z))

        #------

        if radius==None:
            radius = self.radius

        cosmo = cosmology.LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)
        search_radius = radius * cosmo.arcsec_per_kpc_comoving(self.z_BCG).value / 3600

        candidates = catalog_distance(self.catalog['ra'][self.BCG_idx], 
            self.catalog['dec'][self.BCG_idx], self.catalog) < search_radius

        #------
        cluster_members = self.catalog['redshift'][np.logical_and(zmask, candidates)]

        self.mean_z = np.mean(cluster_members)
        self.std_z = np.std(cluster_members)
        self.richness = len(cluster_members)

    def return_statistics(self):
        return self.z_BCG, self.mean_z, self.std_z, self.richness, self.catalog['r'][self.BCG_idx]