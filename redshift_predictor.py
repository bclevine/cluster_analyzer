import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import cm, colors
from helpers import *
import collections

class Predictor:
    def __init__(self, zfile):
        '''
        - zfile: location of redshift file
        - mag_range: initial range of magnitudes
        
        '''
        self.zvals = process_zvals(zfile)
        self.r_data = None
        self.z_data = None

    def update_linelist(self, coltype, mag_range):
        self.linelist = compute_slope(self.zvals, coltype, mag_range)
        return self.linelist
    
    def compute_arrays(self, mag_range=0.5):
        self.gr_slopes = self.update_linelist('gr', mag_range)
        self.rz_slopes = self.update_linelist('rz', mag_range)

        self.gr_red = [x[2] for x in self.gr_slopes]
        self.rz_red = [x[2] for x in self.rz_slopes]

        self.gr_col = [x[1] for x in self.gr_slopes]
        self.rz_col = [x[1] for x in self.rz_slopes]

        self.gr_mags = [x[0] for x in self.gr_slopes]
        self.rz_mags = [x[0] for x in self.rz_slopes]

        self.gr_slope_mask = [x>-10 for x in self.zvals['g-r_slope']]
        self.rz_slope_mask = [x>-10 for x in self.zvals['r-z_slope']]
        self.gr_perp = np.array(-1 * 1/(self.zvals['g-r_slope'][self.gr_slope_mask]))
        self.rz_perp = np.array(-1 * 1/(self.zvals['r-z_slope'][self.rz_slope_mask]))

        self.gr_zvals = self.zvals[self.zvals['r_mag'] != -10]
        self.rz_zvals = self.zvals[self.zvals['z_mag'] != -10]

    def compute_distance(self, color, magnitude, idx, coltype):
        if coltype == 'gr':
            perp = self.gr_perp
            colors = self.gr_col
            mags = self.gr_mags
        elif coltype == 'rz':
            perp = self.rz_perp
            colors = self.rz_col
            mags = self.rz_mags

        x1, y1 = magnitude, color
        if isinstance(color, (collections.Sequence, np.ndarray, pd.Series)):
            x2, y2 = get_alt_point(magnitude, color, values_from_index_list(perp, idx))
            x3, y3 = values_from_index_list(mags, idx), values_from_index_list(colors, idx)
            x4, y4 = values_from_index_list(mags, idx, 1), values_from_index_list(colors, idx, 1)

            xint, yint = find_intersection(x1,y1,x2,y2,x3,y3,x4,y4)
            return np.linalg.norm((xint-x1, yint-y1), axis=0)

        else:
            x2, y2 = get_alt_point(magnitude, color, perp[idx])
            x3, y3 = mags[idx][0], colors[idx][0]
            x4, y4 = mags[idx][1], colors[idx][1]
        
            xint, yint = find_intersection(x1,y1,x2,y2,x3,y3,x4,y4) 
            return np.linalg.norm((xint-x1, yint-y1))

    def color_from_mag(self, magnitude, coltype):
        #coltype either 'gr' or 'rz'
        if coltype == 'gr':
            mag = 'r_mag'
            col = 'g-r_col'
            slope = 'g-r_slope'
            zvals = self.gr_zvals
        elif coltype == 'rz':
            mag = 'z_mag'
            col = 'r-z_col'
            slope = 'r-z_slope'
            zvals = self.rz_zvals
        else:
            raise ValueError("Unknown color type.")

        if isinstance(magnitude, (collections.Sequence, np.ndarray, pd.Series)):
            delta_mags = np.broadcast_to(zvals[mag], (len(magnitude),len(zvals[mag]))) - magnitude[:,None]
            return np.broadcast_to(zvals[col], (len(magnitude),len(zvals[col]))) - (np.broadcast_to(zvals[slope], 
                (len(magnitude),len(zvals[slope]))) * delta_mags)
        else:
            delta_mag = zvals[mag] - magnitude
            return zvals[col] - (zvals[slope] * delta_mag)

    def find_nearest(self, color, magnitude, coltype):
        clist = self.color_from_mag(magnitude, coltype)

        if isinstance(color, (collections.Sequence, np.ndarray, pd.Series)):
            idx = np.abs(clist - color[:,None])
            return np.argpartition(idx, 2, axis=1)[:,:2]
        else:
            idx = np.abs(clist - color)
            return np.argpartition(idx, 2)[:2]

    def nearest_redshift(self, color, magnitude, coltype):
        if isinstance(color, (collections.Sequence, np.ndarray, pd.Series)):
            idx_list = self.find_nearest(color, magnitude, coltype)
            idx1 = idx_list[:,0]
            idx2 = idx_list[:,1]

            d1 = self.compute_distance(color, magnitude, idx1, coltype)
            d2 = self.compute_distance(color, magnitude, idx2, coltype)

            indicies = np.zeros(len(idx_list))
            indicies[d1>d2] = idx2[d1>d2]
            indicies[d1<d2] = idx1[d1<d2]
            indicies = indicies.astype(int)
            
            if coltype == 'gr':
                red = self.gr_red
            elif coltype == 'rz':
                red = self.rz_red

            val_list = np.broadcast_to(red, (len(indicies),len(red)))

            return val_list[np.arange(len(val_list)), indicies], indicies

        else:
            idx1, idx2 = self.find_nearest(color, magnitude, coltype)
            d1 = self.compute_distance(color, magnitude, idx1, coltype)
            d2 = self.compute_distance(color, magnitude, idx2, coltype)
            
            if d1 < d2:
                idx = idx1
            elif d2 < d1:
                idx = idx2
            elif d1 == d2:
                print('Distance is equal.')
                idx = idx1
            
            try:
                if coltype == 'gr':
                    return self.gr_red[idx], idx
                elif coltype == 'rz':
                    return self.rz_red[idx], idx
            except:
                print(color, magnitude)

    def import_test_data(self, fluxg, fluxr, fluxz, mags=False):
        '''
        mags: if True, then fluxg/r/z are interpreted as magnitudes
        '''
        if mags==False:
            self.g_data = flux_to_mag(fluxg)
            self.r_data = flux_to_mag(fluxr)
            self.z_data = flux_to_mag(fluxz)
        elif mags==True:
            self.g_data = fluxg
            self.r_data = fluxr
            self.z_data = fluxz
        self.gr_data = self.g_data - self.r_data
        self.rz_data = self.r_data - self.z_data

    def predict(self, verbose=True):
        self.gr_prediction = self.nearest_redshift(self.gr_data, self.r_data, 'gr')
        self.rz_prediction = self.nearest_redshift(self.rz_data, self.z_data, 'rz')

        if verbose:
            print('g-r: z =', self.gr_prediction[0])
            print('r-z: z =', self.rz_prediction[0])
        
        return self.gr_prediction, self.rz_prediction

    def plot_prediction(self, mrange=0.5):
        plot_pred(self.gr_slopes, self.rz_slopes, self.gr_red, self.rz_red, self.gr_prediction, self.rz_prediction, 
            self.gr_data, self.r_data, self.rz_data, self.z_data, self.zvals, self.gr_slope_mask, self.rz_slope_mask, mrange)

    def plot_lines(self):
        plot_plain(self.gr_slopes, self.rz_slopes, self.gr_red, self.rz_red)

    def predict_from_values(self, r, z, gr, rz, verbose=True, mag_lim=2):
        if isinstance(r, (collections.Sequence, np.ndarray, pd.Series)):
            nan_mask = ~np.isnan([gr, r]).any(axis=0)
            inf_mask = ~np.isinf([gr, r]).any(axis=0)
            total_mask = np.logical_and(nan_mask, inf_mask)

            self.gr_prediction = np.full(len(r), np.nan)
            self.gr_idxs = np.full(len(r), np.nan)
            reds, idxs = self.nearest_redshift(gr[total_mask], r[total_mask], 'gr')
            self.gr_prediction[total_mask] = reds
            self.gr_idxs[total_mask] = idxs

            nan_mask = ~np.isnan([rz, z]).any(axis=0)
            inf_mask = ~np.isinf([rz, z]).any(axis=0)
            total_mask = np.logical_and(nan_mask, inf_mask)
            
            self.rz_prediction = np.full(len(z), np.nan)
            self.rz_idxs = np.full(len(z), np.nan)
            reds, idxs = self.nearest_redshift(rz[total_mask], z[total_mask], 'rz')
            self.rz_prediction[total_mask] = reds
            self.rz_idxs[total_mask] = idxs

            #mag cut
            self.rz_prediction[~(z.reset_index(drop=True) < (self.zvals['z_mag'][np.nan_to_num(self.rz_idxs+np.where(self.zvals['z_mag'] != -10)[0][0], nan=0)]+
                mag_lim).reset_index(drop=True))] = np.nan
            self.gr_prediction[~(r.reset_index(drop=True) < (self.zvals['r_mag'][np.nan_to_num(self.gr_idxs, nan=len(self.zvals)-1)]+
                mag_lim).reset_index(drop=True))] = np.nan

            return self.gr_prediction, self.rz_prediction

        else:
            self.r_data, self.z_data = r, z
            self.gr_data, self.rz_data = gr, rz
            if np.isnan(np.array([gr, r]).astype(float)).any():
                self.gr_prediction = None, None
            elif np.isinf(np.array([gr, r]).astype(float)).any():
                self.gr_prediction = None, None
            else:
                self.gr_prediction = self.nearest_redshift(gr, r, 'gr')

            if np.isnan(np.array([rz, z]).astype(float)).any():
                self.rz_prediction = None, None
            elif np.isinf(np.array([rz, z]).astype(float)).any():
                self.rz_prediction = None, None
            else:  
                self.rz_prediction = self.nearest_redshift(rz, z, 'rz')

            if verbose:
                print('g-r: z =', self.gr_prediction[0])
                print('r-z: z =', self.rz_prediction[0])
            
            return self.gr_prediction, self.rz_prediction
