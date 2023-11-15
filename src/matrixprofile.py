import numpy as np
from joblib import Parallel,delayed
import distance as distance
import itertools as it 
from functools import partial
from pathlib import Path
import subprocess 
import os

import warnings
warnings.filterwarnings('ignore')


class MatrixProfile(object): 

    def __init__(self,n_patterns:int,wlen:int,distance_name:str,distance_params = dict(),radius_ratio = 3,n_jobs = 1) -> None:
        """Initialization

        Args:
            n_patterns (int): Number of neighbors
            wlen (int): Window length
            distance_name (str): name of the distance
            distance_params (_type_, optional): additional distance parameters. Defaults to dict().
            radius_ratio (float): radius as a ratio of min_dist. 
            n_jobs (int, optional): number of processes. Defaults to 1.
        """
        self.n_patterns = n_patterns
        self.radius_ratio = radius_ratio
        self.wlen = wlen
        self.distance_name = distance_name
        self.distance_params = distance_params
        self.n_jobs = n_jobs

    def _search_neighbors(self,idx:int,line:np.ndarray)-> tuple: 
        """Find index and distance value of the non overlapping nearest neighbors under a radius.

        Args:
            idx (int): index of the considerded line in the crossdistance matrix
            line (np.ndarray): line of the crossdistance matrix. shape: (n_sample,)

        Returns:
            tuple: neighbor index np.ndarray, neighbor distance np.ndarray
        """

        #initilization
        neighbors = []
        dists = []
        idxs = np.arange(self.mdim_)
        remove_idx = np.arange(max(0,idx-self.wlen+1),min(self.mdim_,idx+self.wlen))
        idxs = np.delete(idxs,remove_idx)
        line = np.delete(line,remove_idx)

        #search loop
        radius = np.min(line)*self.radius_ratio
        t_distance = np.min(line)
        while t_distance < radius:
            try: 
                #local next neighbor
                t_idx = np.argmin(line)
                neighbors.append(idxs[t_idx])
                dists.append(line[t_idx])

                #remove window
                remove_idx = np.arange(max(0,t_idx-self.wlen+1),min(len(line),t_idx+self.wlen))
                idxs = np.delete(idxs,remove_idx)
                line = np.delete(line,remove_idx)

                t_distance = dists[-1]
            except: 
                break
            
        return neighbors,dists

    def _elementary_profile(self,start:int,end:int)->tuple:
        """Find elementary profile of a chunk of successive lines of the crossdistance matrix

        Args:
            start (int): chunk start
            end (int): chunck end

        Returns:
            tuple: neighborhood count, neighborhood std
        """
        #initialization
        neighbors =[]
        dists = []
        line = self.distance_.first_line(start)
        mask = np.arange(max(0,start-self.wlen+1), min(self.mdim_,start+self.wlen))
        line[mask] = np.inf
        t_idx = np.argmin(line)
        t_dist = line[t_idx]
        neighbors.append(t_idx)
        dists.append(t_dist)
        
        #main loop
        for i in range(start+1,end): 
            line = self.distance_.next_line()
            mask = np.arange(max(0,i-self.wlen+1), min(self.mdim_,i+self.wlen))
            line[mask] = np.inf
            t_idx = np.argmin(line)
            t_dist = line[t_idx]
            neighbors.append(t_idx)
            dists.append(t_dist)
        return neighbors,dists

    def profile_(self)->None: 

        #divide the signal accordingly to the number of jobs
        set_idxs = np.linspace(0,self.mdim_,self.n_jobs+1,dtype=int)
        set_idxs = np.vstack((set_idxs[:-1],set_idxs[1:])).T

        # set the parrallel computation
        results = Parallel(n_jobs=self.n_jobs, prefer="processes")(
            delayed(self._elementary_profile)(*set_idx)for set_idx in set_idxs
        )

        idxs,dists = list(zip(*results))

        self.idxs_ = np.hstack(idxs)
        self.dists_ = np.hstack(dists)
        return self

    def find_patterns_(self): 
        profile = self.dists_.copy()
        mask = []
        patterns = []

        for _ in np.arange(self.n_patterns): 
            min_idx = np.argmin(profile)
            if profile[min_idx]==np.inf: 
                break
            line = self.distance_.first_line(min_idx)
            line[mask] = np.inf
            p_idxs,dists = self._search_neighbors(min_idx,line)
            p_idxs = np.hstack((np.array([min_idx]),p_idxs))
            patterns.append(p_idxs)
            mask += np.hstack([np.arange(max(0,idx-self.wlen+1),min(self.mdim_,idx+self.wlen)) for idx in p_idxs]).astype(int).tolist()
            profile[mask] = np.inf
        
        self.patterns_ = patterns

    def fit(self,signal:np.ndarray)->None:
        """Compute the best patterns

        Args:
            signal (np.ndarray): Univariate time-series, shape: (L,)
        """
        
        #initialisation
        self.signal_ = signal
        self.mdim_ = len(signal)-self.wlen+1 
        self.distance_ = getattr(distance,self.distance_name)(self.wlen,**self.distance_params)
        self.distance_.fit(signal)

        #Compute neighborhood
        self.profile_()
        #find patterns
        self.find_patterns_()

        return self

    @property
    def prediction_mask_(self):
        mask = np.zeros((self.n_patterns,self.signal_.shape[0]))
        for i,p_idxs in enumerate(self.patterns_):
            for idx in p_idxs.astype(int):
                mask[i,idx:idx+self.wlen]=1 
        return mask