#IMPORTS
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from astropy import cosmology
import matplotlib.image as mpimg
import os
import urllib.request as urlib
import socket
from astropy.io import fits
from helpers import *

cosmo = cosmology.WMAP9

#SET WHICH BITMASKS TO REMOVE (see https://www.legacysurvey.org/dr9/bitmasks/)
#You can also consider using flag 8, but it tends to remove a whole lot of objects...
bitflags = [1,9]

class Cluster:
    def __init__(self, data, RA, DEC, Predictor, radius=500, ctol=0.2, del_masks=True, uncertainty_floor=0.03):
        """Initialization for Cluster object

        Args:
            data (astropy.fits data table): Catalog cutout for the cluster.
            RA (float): Initial guess for RA.
            DEC (float): Initial guess for DEC.
            Predictor (RS Predictor object): Pre-initialized redshift Predictor object.
            radius (float, optional): Search radius of cluster in kpc. Defaults to 500.
            ctol (float, optional): Color separation tolerance for BCG finder. Defaults to 0.2.
            del_masks (bool, optional): Whether or not to apply bitwise masking. Defaults to True.
            uncertainty_floor (float, optional): Uncertainty floor for red-sequence scatter. Defaults to 0.03.
        """        
        
        #IMPORT FLUX DATA
        gvar = (1/np.sqrt(data['flux_ivar_g']))
        rvar = (1/np.sqrt(data['flux_ivar_r']))
        zvar = (1/np.sqrt(data['flux_ivar_z']))

        fluxg = data['flux_g']/data['mw_transmission_g'].byteswap().newbyteorder()
        fluxgmax = (data['flux_g']+gvar)/data['mw_transmission_g'].byteswap().newbyteorder()
        fluxgmin = (data['flux_g']-gvar)/data['mw_transmission_g'].byteswap().newbyteorder()
        fluxr = data['flux_r']/data['mw_transmission_r'].byteswap().newbyteorder()
        fluxrmax = (data['flux_r']+rvar)/data['mw_transmission_r'].byteswap().newbyteorder()
        fluxrmin = (data['flux_r']-rvar)/data['mw_transmission_r'].byteswap().newbyteorder()
        fluxz = data['flux_z']/data['mw_transmission_z'].byteswap().newbyteorder()
        fluxzmax = (data['flux_z']+zvar)/data['mw_transmission_z'].byteswap().newbyteorder()
        fluxzmin = (data['flux_z']-zvar)/data['mw_transmission_z'].byteswap().newbyteorder()

        #CREATE DATAFRAME
        self.catalog = pd.DataFrame({'ra':data['ra'].byteswap().newbyteorder(), 
            'dec':data['dec'].byteswap().newbyteorder(),
            'g':flux_to_mag(fluxg),
            'r':flux_to_mag(fluxr),
            'z':flux_to_mag(fluxz),

            'g_e':add_quadrature(calc_magnitude_error(fluxg, fluxgmax, fluxgmin), uncertainty_floor),
            'r_e':add_quadrature(calc_magnitude_error(fluxr, fluxrmax, fluxrmin), uncertainty_floor),
            'z_e':add_quadrature(calc_magnitude_error(fluxz, fluxzmax, fluxzmin), uncertainty_floor)
            })
        
        #COMPUTE COLORS
        self.catalog['gr'] = self.catalog['g'] - self.catalog['r']
        self.catalog['rz'] = self.catalog['r'] - self.catalog['z']

        #BITWISE MASKING
        if del_masks:
            self.flags = ((((data['maskbits'][:,None] & (1 << np.arange(14)))) > 0).astype(int))[:,bitflags] 
            self.catalog = self.catalog[np.all((self.flags == 0), axis=1)].reset_index(drop=True)
        
        #SAVE PREFERENCES
        self.Predictor = Predictor
        self.radius = radius
        self.ctol = ctol
        self.RA = RA
        self.DEC = DEC
        self.uncertainty_floor = uncertainty_floor

    def count_all(self):
        """Utility function for counting objects in catalog

        Returns:
            int: length of catalog
        """        
        
        return len(self.catalog.index)

    def identify_BCG(self, radius=None, ctol=None, iterate=True, use_given_coords=False):
        """Finds the BCG of the cluster

        Args:
            radius (float, optional): Search radius of cluster in kpc. Defaults to whatever was set in Cluster initialization (default 500).
            ctol (float, optional): Color separation tolerance for BCG finder. Defaults to whatever was set in Cluster initialization (default 0.2).
            iterate (bool, optional): Should the function iterate more than once? Defaults to True.
            use_given_coords (bool, optional): Should the function just pick the object at the initial RA/DEC guess? Defaults to False.

        Returns:
            int: index of BCG
            (also updates  relevant values inside the Cluster object; ie. BCG_idx and z_BCG)
        """   
        
        #LOAD IN DEFAULTS FOR RADIUS AND CTOL
        if radius==None:
            radius = self.radius
        if ctol==None:
            ctol = self.ctol

        #IDENTIFY INITIAL GUESS BY FINDING NEAREST OBJECT TO GUESSED COORDINATES
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
        
        #SAVE INITIAL REDSHIFT
        n = np.array(self.Predictor.predict_from_values(self.catalog['r'][self.BCG_idx], 
            self.catalog['z'][self.BCG_idx], self.catalog['gr'][self.BCG_idx], 
            self.catalog['rz'][self.BCG_idx], False), dtype=float)[:,0]
        self.initial_z_BCG = combine_redshifts(n[0], n[1])

        #IF USING GIVEN COORDINATES, UPDATE ALL VALUES AND RETURN BCG_idx
        if use_given_coords:
            n = np.array(self.Predictor.predict_from_values(self.catalog['r'][self.BCG_idx], 
                self.catalog['z'][self.BCG_idx], self.catalog['gr'][self.BCG_idx], 
                self.catalog['rz'][self.BCG_idx], False), dtype=float)[:,0]
            self.z_BCG = combine_redshifts(n[0], n[1])
            return self.BCG_idx

        #CHECK ADDITIONAL CANDIDATES
        zred = np.nanmean(np.array(self.Predictor.predict_from_values(self.catalog['r'][BCG_idx], self.catalog['z'][BCG_idx],
            self.catalog['gr'][BCG_idx], self.catalog['rz'][BCG_idx], False), dtype=float), axis=0)[0]

        search_radius = radius * cosmo.arcsec_per_kpc_proper(zred).value / 3600

        found = False
        iterations = 0

        #iterate over all candidates
        while found==False:
            candidates = [catalog_distance(self.catalog['ra'][BCG_idx], 
                self.catalog['dec'][BCG_idx], self.catalog) < search_radius]
            if not np.array(candidates).any():
                found = True
                break
            else:
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

                        #interation limit of 100
                        if iterations >= 100:
                            found = True

                        break
            found = True
        
        self.BCG_idx = BCG_idx

        #UPDATE ALL VALUES AND RETURN BCG_idx
        n = np.array(self.Predictor.predict_from_values(self.catalog['r'][self.BCG_idx], 
            self.catalog['z'][self.BCG_idx], self.catalog['gr'][self.BCG_idx], 
            self.catalog['rz'][self.BCG_idx], False), dtype=float)[:,0]
        self.z_BCG = combine_redshifts(n[0], n[1])
        return self.BCG_idx

    def plot_BCG(self, catalog_width=0.03, img=None, pxscale=0.262, imgwidth=250, xscale=1, yscale=1):
        """Produces a plot of the BCG

        Args:
            catalog_width (float, optional): Angular scale of catalog, in arcsec. Defaults to 0.03.
            img (str, optional): Path to jpeg image of cluster to show in background. Defaults to None.
            pxscale (float, optional): Pixel scale of jpeg image (survey-dependent, this should never really change). Defaults to 0.262.
            imgwidth (int, optional): Pixel width of cutout, assuming a square. Defaults to 250.
            xscale (int, optional): Horizontal stretch factor of image based on declination. It should be cos(dec), but I'm lazy. Defaults to 1.
            yscale (int, optional): Vertixal stretch factor of image. You should never have to use this. Defaults to 1.
        """     
           
        cmp = plt.cm.get_cmap('RdYlBu')
        
        #IF THERE IS NO BACKGROUND IMAGE...
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
            
        #APPLY BACKGROUND IMAGE
        else:
            catrange = (pxscale*imgwidth/2)/3600
            ramask = [(self.catalog['ra'] < self.RA+catrange) and (self.catalog['ra'] > self.RA-catrange)]
            decmask = [(self.catalog['dec'] < self.DEC+catrange) and (self.catalog['dec'] > self.DEC-catrange)]
            catmask = np.logical_and(ramask, decmask).flatten()
            ra = self.catalog['ra']; dec = self.catalog['dec']

            #RENORMALIZE COORDINATES
            ra_px = np.abs(((ra-self.RA)*xscale-catrange)*250/(2*catrange))
            dec_px = np.abs(((dec-self.DEC)*yscale-catrange)*250/(2*catrange))
            
            plt.figure(figsize=(8,6))
            plt.imshow(mpimg.imread(img), zorder=0)
            plt.scatter(ra_px[catmask], dec_px[catmask], s=10, c=self.catalog['r'][catmask], 
                vmax=np.median(self.catalog['r'].dropna()), vmin=self.catalog['r'][self.BCG_idx], cmap=cmp, zorder=10)
            plt.scatter(ra_px[self.BCG_idx], dec_px[self.BCG_idx], s=150, marker='o', alpha=.4, zorder=11)
            plt.colorbar()
            plt.show()
            
    def identify_cluster(self, algorithm='color', radius=None, ctol=None, ztol=0.2):
        """Identify cluster members (DEPRACATED)

        Args:
            algorithm (str, optional): Which algorithm to use. Defaults to 'color'.
                Options:
                    - 'color' (faster, compares colors within ctol of BCG)
                    - 'redshift' (slower, compares redshifts within ztol of BCG)
            radius (float, optional): Search radius of cluster in kpc. Defaults to whatever was set in Cluster initialization (default 500).
            ctol (float, optional): Color separation tolerance. Defaults to whatever was set in Cluster initialization (default 0.2).
            ztol (float, optional): Redshift separation tolerance. Defaults to 0.2.

        Returns:
            list: indicies of cluster members
            
        Notes:
            This function is depracated and should not be used. Use calc_richness to find cluster members instead.
        """        

        if radius==None:
            radius = self.radius
        if ctol==None:
            ctol = self.ctol

        search_radius = radius * cosmo.arcsec_per_kpc_proper(self.z_BCG).value / 3600

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
        """Plot the cluster members. self.cluster_members must exist (ie. run a cluster finding function beforehand).

        Args:
            catalog_width (float, optional): Angular scale of the catalog, in arcsec. Defaults to 0.03.
        """      
          
        plt.figure(figsize=(8,6))
        plt.scatter(self.catalog['ra'], self.catalog['dec'], s=10, c='red')
        plt.scatter(self.catalog['ra'][self.cluster_members], self.catalog['dec'][self.cluster_members], s=30, c='blue')
        plt.scatter(self.catalog['ra'][self.BCG_idx], self.catalog['dec'][self.BCG_idx], s=50, c='green')
        plt.xlim(self.RA+catalog_width, self.RA-catalog_width)
        plt.ylim(self.DEC-catalog_width, self.DEC+catalog_width)
        plt.show()

    def calc_redshifts(self, radius=None, large=False, full_cat=False, mag_lim=2):
        """Calculate redshifts objects in the catalog

        Args:
            radius (float, optional): Search radius of cluster in kpc. Defaults to whatever was set in Cluster initialization (default 500).
            large (bool, optional): Is this a large (background) catalog? Defaults to False.
            full_cat (bool, optional): Should the function calculate for every single object in the catalog (ignore the radius)? Defaults to False.
            mag_lim (int, optional): Magnitude limit for M* —> how faint do we count? Defaults to 2.

        Returns:
            None
        """        
        
        #SMALL CATALOG, AND ONLY OBJECTS IN RADIUS
        if not (large or full_cat):
            #CONVERT RADIUS TO ANGLE MEASUREMENT
            if radius==None:
                radius = self.radius
            search_radius = radius * cosmo.arcsec_per_kpc_proper(self.z_BCG).value / 3600

            #SELECT OBJECTS WITHIN RADIUS
            candidates = [catalog_distance(self.catalog['ra'][self.BCG_idx], 
                self.catalog['dec'][self.BCG_idx], self.catalog) < search_radius]
            if not np.array(candidates).any():
                return None
            candidate_idx = np.array(list(zip(*np.argwhere(candidates)))[1])

            #COMPUTE AND SAVE REDSHIFTS
            redshifts = self.Predictor.predict_from_values(self.catalog['r'][candidate_idx], 
                self.catalog['z'][candidate_idx], self.catalog['gr'][candidate_idx], self.catalog['rz'][candidate_idx], False, mag_lim=mag_lim)
                    
            self.catalog['gr_redshift'] = np.NaN
            self.catalog['rz_redshift'] = np.NaN
            self.catalog['gr_redshift'][candidate_idx] = redshifts[0].astype('float64') 
            self.catalog['rz_redshift'][candidate_idx] = redshifts[1].astype('float64') 
            
            #COMPUTE AND SAVE REDSHIFT UNCERTAINTIES
            redshifts_max, redshifts_min = predict_magnitudes_max_min(self, candidate_idx)

            self.catalog['gr_redshift+'] = np.NaN
            self.catalog['rz_redshift+'] = np.NaN
            self.catalog['gr_redshift+'][candidate_idx] = redshifts_max[0].astype('float64') 
            self.catalog['rz_redshift+'][candidate_idx] = redshifts_max[1].astype('float64') 

            self.catalog['gr_redshift-'] = np.NaN
            self.catalog['rz_redshift-'] = np.NaN
            self.catalog['gr_redshift-'][candidate_idx] = redshifts_min[0].astype('float64') 
            self.catalog['rz_redshift-'][candidate_idx] = redshifts_min[1].astype('float64') 

        #SMALL CATALOG, AND ALL OBJECTS
        elif full_cat:
            #COMPUTE AND SAVE REDSHIFTS
            redshifts = self.Predictor.predict_from_values(self.catalog['r'], 
                self.catalog['z'], self.catalog['gr'], self.catalog['rz'], False, mag_lim=mag_lim)
                    
            self.catalog['gr_redshift'] = np.NaN
            self.catalog['rz_redshift'] = np.NaN
            self.catalog['gr_redshift'] = redshifts[0].astype('float64') 
            self.catalog['rz_redshift'] = redshifts[1].astype('float64')
            
            #COMPUTE AND SAVE REDSHIFT UNCERTAINTIES
            redshifts_max, redshifts_min = predict_magnitudes_max_min(self)

            self.catalog['gr_redshift+'] = np.NaN
            self.catalog['rz_redshift+'] = np.NaN
            self.catalog['gr_redshift+'] = redshifts_max[0].astype('float64') 
            self.catalog['rz_redshift+'] = redshifts_max[1].astype('float64') 

            self.catalog['gr_redshift-'] = np.NaN
            self.catalog['rz_redshift-'] = np.NaN
            self.catalog['gr_redshift-'] = redshifts_min[0].astype('float64') 
            self.catalog['rz_redshift-'] = redshifts_min[1].astype('float64') 

        #LARGE CATALOG
        elif large:
            #ASSERT THAT LARGE CATALOG EXISTS
            try:
                self.large_catalog
            except NameError:
                print('Error: Import large catalog first.')
                return None

            #CONVERT RADIUS TO ANGLE MEASUREMENT
            if radius==None:
                radius = 5000
            search_radius = radius * cosmo.arcsec_per_kpc_proper(self.z_BCG).value / 3600

            #SELECT OBJECTS WITHIN RADIUS
            candidates = [catalog_distance(self.catalog['ra'][self.BCG_idx], 
                self.catalog['dec'][self.BCG_idx], self.large_catalog) > search_radius]
            candidate_idx = np.array(list(zip(*np.argwhere(candidates)))[1])

            #COMPUTE AND SAVE REDSHIFTS
            redshifts = self.Predictor.predict_from_values(self.large_catalog['r'][candidate_idx], 
                self.large_catalog['z'][candidate_idx], self.large_catalog['gr'][candidate_idx], 
                self.large_catalog['rz'][candidate_idx], False, mag_lim=mag_lim)

            self.large_catalog['gr_redshift'] = np.NaN
            self.large_catalog['rz_redshift'] = np.NaN
            self.large_catalog['gr_redshift'][candidate_idx] = redshifts[0].astype('float64') 
            self.large_catalog['rz_redshift'][candidate_idx] = redshifts[1].astype('float64')

        #COMBINE REDSHIFTS, AND RECALCULATE z_BCG
        self.catalog['redshift'] = combine_redshifts(self.catalog)
        self.z_BCG = self.catalog['redshift'][self.BCG_idx]


    def download_surroundings(self, directory='DECALS Cutout Grabber/catalogs.nosync', verbose=True, delete=False, angular_scale=None, physical_scale=None):
        """Download background for the catalog

        Args:
            directory (str, optional): Where should the catalog be saved to?. Defaults to 'DECALS Cutout Grabber/catalogs.nosync'.
            verbose (bool, optional): Verbose? Defaults to True.
            delete (bool, optional): Should we delete the fits file after loading the data? Defaults to False.
            angular_scale (float, optional): Angular size in degrees. Defaults to None.
            physical_scale (float, optional): Physical size in kpc. Defaults to None.
                Note: ONLY give angular_scale or physical_scale, NOT BOTH

        Returns:
            None
        """        
        
        ra = self.catalog['ra'][self.BCG_idx]
        dec = self.catalog['dec'][self.BCG_idx]
        outname = '/large_'+str(ra)+'_'+str(dec)+'.fits'
        path = os.getcwd()+'/'+directory

        #CALCULATE SCALE
        if angular_scale==None:
            if physical_scale==None:
                size = 10000 * cosmo.arcsec_per_kpc_proper(self.z_BCG).value / 3600
            else: 
                size = physical_scale * cosmo.arcsec_per_kpc_proper(self.z_BCG).value / 3600
        else:
            size = angular_scale
        
        self.large_scale = size

        #CHECK IF FILE ALREADY EXISTS
        if not os.path.isfile(path+outname):
            url = 'https://www.legacysurvey.org/viewer/ls-dr9/cat.fits?ralo={}&rahi={}&declo={}&dechi={}'

            dec_min = dec-size; dec_max = dec+size
            size = size/np.cos(np.radians(dec))
            ra_min = ra-size; ra_max = ra+size
            
            #Change the below value to increase timeout maximum
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

        #LOAD CATALOG
        hdul = fits.open(directory+outname)
        data = hdul[1].data
        hdul.close()

        #SAVE FLUX DATA
        self.large_catalog = pd.DataFrame({'ra':data['ra'].byteswap().newbyteorder(), 
            'dec':data['dec'].byteswap().newbyteorder(),
            'g':flux_to_mag(data['flux_g']/data['mw_transmission_g'].byteswap().newbyteorder()),
            'r':flux_to_mag(data['flux_r']/data['mw_transmission_r'].byteswap().newbyteorder()),
            'z':flux_to_mag(data['flux_z']/data['mw_transmission_z'].byteswap().newbyteorder()),
            })
        self.large_catalog['gr'] = self.large_catalog['g'] - self.large_catalog['r']
        self.large_catalog['rz'] = self.large_catalog['r'] - self.large_catalog['z']

        #DELETE FITS FILE
        if delete:
            os.remove(path+outname)


    def plot_distribution(self, type, bins=50, range=[0,1.5]):
        """Plot the histogram of cluster vs. background objects in bins of redshift

        Args:
            type (str): Either 'gr' or 'rz', corresponding to redshift
            bins (int, optional): How many redshift bins to use. Defaults to 50.
            range (list, optional): Min and max redshift values. Defaults to [0,1.5].
        """        

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


    def calc_background(self, bins=50, range=[0,1.5]):
        """Calculate the background histogram for the cluster

        Args:
            bins (int, optional): How many redshift bins to use. Defaults to 50.
            range (list, optional): Min and max redshift values. Defaults to [0,1.5].

        Returns:
            list: Histogram showing number of cluster members in each redshift bin.
        """        
        self.catalog['redshift'] = combine_redshifts(self.catalog)  
        self.z_BCG = self.catalog['redshift'][self.BCG_idx]
        
        cluster_dist = np.histogram(self.catalog['redshift'].dropna(), bins=bins, density=False, range=range)
        self.background_dist = cluster_dist
        if cluster_dist==None:
            return cluster_dist
        else:
            return cluster_dist[0]


    def calc_richness(self, bins=50, range=[0,1.5], radius=None, debug=False, comparison_redshift=None, comparison_tolerance=0.05, use_uncertainty=False, sigma=3):
        """Calculate richness of the cluster

        Args:
            bins (int, optional): How many redshift bins to use. Defaults to 50.
            range (list, optional): Min and max redshift values. Defaults to [0,1.5].
            radius (float, optional): Search radius of cluster in kpc. Defaults to whatever was set in Cluster initialization (default 500).
            debug (bool, optional): Return various helpful additional values. Defaults to False.
            comparison_redshift (str OR float, optional): Don't use uncertainties and instead set value of cluster redshift. 
                                                          Can be a float, or 'z_BCG' to use z_BCG. Defaults to None.
            comparison_tolerance (float, optional): Redshift tolerance if using a comparison redshift. Defaults to 0.05.
            use_uncertainty (bool, optional): Whether or not to use uncertainties. Defaults to False.
            sigma (int, optional): Sigma, if using uncertainties. Defaults to 3.

        Returns:
            tuple: None if debug=False; various values if debug=True
        """        
        
        #CALCULATE REDSHIFT AND UPDATE z_BCG
        self.catalog['redshift'] = combine_redshifts(self.catalog)
        self.z_BCG = self.catalog['redshift'][self.BCG_idx]

        if radius==None: 
            radius=self.radius

        #IF USING UNCERTAINTIES...
        if use_uncertainty:
            if comparison_redshift == None:
                #CONVERT SEARCH RADIUS TO ANGLE
                search_radius = radius * cosmo.arcsec_per_kpc_proper(self.z_BCG).value / 3600
                candidates = catalog_distance(self.catalog['ra'][self.BCG_idx], 
                    self.catalog['dec'][self.BCG_idx], self.catalog) < search_radius

                #COMPUTE UNCERTAINTY REDSHIFTS
                self.catalog['redshift+'] = combine_redshifts(self.catalog['gr_redshift+'], self.catalog['rz_redshift+'])
                self.catalog['redshift-'] = combine_redshifts(self.catalog['gr_redshift-'], self.catalog['rz_redshift-'])

                maxvals = np.nanmax([self.catalog['redshift'], self.catalog['redshift+'], self.catalog['redshift-']], axis=0)
                minvals = np.nanmin([self.catalog['redshift'], self.catalog['redshift+'], self.catalog['redshift-']], axis=0)
                medians = np.nanmedian([self.catalog['redshift'], self.catalog['redshift+'], self.catalog['redshift-']], axis=0)

                #COMPUTE STANDARD DEVIATIONS
                sd_up = maxvals - medians
                sd_down = medians - minvals
                sd_up[sd_up == 0] = self.uncertainty_floor
                sd_down[sd_down == 0] = self.uncertainty_floor

                maxmask = (medians + (sigma*sd_up)) > self.z_BCG
                minmask = (medians - (sigma*sd_down)) < self.z_BCG

                #FIND CLUSTER MEMBERS AND SAVE VALUES
                cluster_members = self.catalog['redshift'][np.logical_and(np.logical_and(maxmask, minmask), candidates)]
                self.mean_z = np.mean(cluster_members)
                self.std_z = np.std(cluster_members)
                self.richness = len(cluster_members)
                
                if debug:
                    self.mean_z = np.mean(cluster_members)
                    self.std_z = np.std(cluster_members)
                    self.richness = len(cluster_members)
                    return cluster_members

                return None
            
            #IF WE ALSO HAVE A COMPARISON REDSHIFT...
            elif comparison_redshift != None:
                #SAME CALCULATION, BUT USE COMPARISON REDSHIFT 
                if comparison_redshift == 'z_BCG':
                    comparison_redshift = self.z_BCG

                #CONVERT SEARCH RADIUS TO ANGLE
                search_radius = radius * cosmo.arcsec_per_kpc_proper(comparison_redshift).value / 3600
                candidates = catalog_distance(self.catalog['ra'][self.BCG_idx], 
                    self.catalog['dec'][self.BCG_idx], self.catalog) < search_radius

                #COMPUTE UNCERTAINTY REDSHIFTS
                self.catalog['redshift+'] = combine_redshifts(self.catalog['gr_redshift+'], self.catalog['rz_redshift+'])
                self.catalog['redshift-'] = combine_redshifts(self.catalog['gr_redshift-'], self.catalog['rz_redshift-'])

                maxvals = np.nanmax([self.catalog['redshift'], self.catalog['redshift+'], self.catalog['redshift-']], axis=0)
                minvals = np.nanmin([self.catalog['redshift'], self.catalog['redshift+'], self.catalog['redshift-']], axis=0)
                medians = np.nanmedian([self.catalog['redshift'], self.catalog['redshift+'], self.catalog['redshift-']], axis=0)

                #COMPUTE STANDARD DEVIATIONS
                sd_up = maxvals - medians
                sd_down = medians - minvals
                sd_up[sd_up == 0] = self.uncertainty_floor
                sd_down[sd_down == 0] = self.uncertainty_floor

                maxmask = (medians + (sigma*sd_up)) > comparison_redshift
                minmask = (medians - (sigma*sd_down)) < comparison_redshift

                #FIND CLUSTER MEMBERS AND SAVE VALUES
                cluster_members = self.catalog['redshift'][np.logical_and(np.logical_and(maxmask, minmask), candidates)]
                self.mean_z = np.mean(cluster_members)
                self.std_z = np.std(cluster_members)
                self.richness = len(cluster_members)

                if debug:
                    self.mean_z = np.mean(cluster_members)
                    self.std_z = np.std(cluster_members)
                    self.richness = len(cluster_members)
                    return cluster_members
                
                return None

        #IF WE HAVE A COMPARISON REDSHIFT AND ARE NOT USING UNCERTAINTIES...
        if comparison_redshift != None:
            if comparison_redshift == 'z_BCG':
                comparison_redshift = self.z_BCG

            #CONVERT RADIUS TO ANGLE
            search_radius = radius * cosmo.arcsec_per_kpc_proper(comparison_redshift).value / 3600
            candidates = catalog_distance(self.catalog['ra'][self.BCG_idx], 
                self.catalog['dec'][self.BCG_idx], self.catalog) < search_radius
            
            #DEFINE CANDIDATES BASED ON COMPARISON TOLERANCE
            zmask = np.abs((self.catalog['redshift'] - comparison_redshift)) < comparison_tolerance

            #FIND CLUSTER MEMBERS AND SAVE VALUES
            cluster_members = self.catalog['redshift'][np.logical_and(zmask, candidates)]
            self.mean_z = np.mean(cluster_members)
            self.std_z = np.std(cluster_members)
            self.richness = len(cluster_members)

            if debug:
                self.mean_z = np.mean(cluster_members)
                self.std_z = np.std(cluster_members)
                self.richness = len(cluster_members)
                return cluster_members
            
            return None

        #----DEPRACATED---- 
        #Note: you can still get this to run if you don't define comparison_redshift or use_uncertainty. But it is depracated.
        #CALC REDSHIFT FOR LARGE CATALOG
        large_redshift = combine_redshifts(self.large_catalog)

        #Calculate overall redshift distribution for cluster and background
        cluster_dist = np.histogram(self.catalog['redshift'].dropna(), bins=bins, density=False, range=range)
        normal_dist = np.histogram(large_redshift[~np.isnan(large_redshift)], bins=bins, density=False, range=range)

        #Normalize distributions over cluster area
        search_radius = self.radius * cosmo.arcsec_per_kpc_proper(self.z_BCG).value / 3600
        cluster_area = calc_area(2*search_radius)
        normal_area = calc_area(2*self.large_scale) - cluster_area
        normal_dist = (normal_dist / normal_area) * cluster_area

        #Find difference
        proportion_dist = cluster_dist[0] - normal_dist[0]
        
        #Identify redshift range of cluster
        BCG_ind = np.digitize(self.z_BCG, cluster_dist[1]) - 1
        min_idx = BCG_ind
        max_idx = BCG_ind

        while proportion_dist[min_idx - 1] > .1:
            min_idx = min_idx - 1
            
        while proportion_dist[max_idx + 1] > .1:
            max_idx = max_idx + 1

        min_z = cluster_dist[1][min_idx]
        max_z = cluster_dist[1][max_idx+1]

        zmask = np.logical_and((self.catalog['redshift'] > min_z), (self.catalog['redshift'] < max_z))

        #------

        search_radius = radius * cosmo.arcsec_per_kpc_proper(self.z_BCG).value / 3600

        candidates = catalog_distance(self.catalog['ra'][self.BCG_idx], 
            self.catalog['dec'][self.BCG_idx], self.catalog) < search_radius

        #------
        
        cluster_members = self.catalog['redshift'][np.logical_and(zmask, candidates)]
        self.mean_z = np.mean(cluster_members)
        self.std_z = np.std(cluster_members)
        self.richness = len(cluster_members)

        if debug:
            return cluster_dist, normal_dist, min_z, max_z


    def return_statistics(self):
        """Return statistics of the cluster (requires BCG finder and richness algorithm to be run first)

        Returns:
            z_BCG (float): Redshift of BCG.
            mean_z (float): Mean redshift of all cluster members.
            std_z (float): Standard deviation of redshift for all cluster members.
            richness (float): Richness of cluster.
            r (float): r-Magnitude of BCG.
        """        
        
        return self.z_BCG, self.mean_z, self.std_z, self.richness, self.catalog['r'][self.BCG_idx]