import numpy as np

class NormalizedEuclidean(object): #Verifed

    def __init__(self,wlen:int,**kwargs)->None:
        """Initialization

        Args:
            wlen (int): window length
        """
        self.wlen = wlen

    def fit(self,signal:np.ndarray):
        """Compute the elementary components of the first line of the crossdistance matrix. 

        Args:
            signal (np.ndarray): signal. shape (n_samples,n_dimension)
        """
        self.signal_ =  signal
        self.first_basic_ = self._first_elemntary()

    def _first_elemntary(self)->np.ndarray:
        """Compute the dotproduct between subsequence of the first line of the crossdistance matrix. 
        Compute the mean and std for each subsequence. 
        """
        self.first_dot_product_ = np.convolve(self.signal_[:self.wlen][::-1],self.signal_,'valid')

        means = np.zeros(len(self.signal_)+1)
        means[1:] = np.cumsum(self.signal_)/self.wlen
        self.means_ = means[self.wlen:]-means[:-self.wlen]

        stds = np.zeros(len(self.signal_)+1)
        stds[1:] =np.cumsum(self.signal_**2)/self.wlen
        stds = stds[self.wlen:]-stds[:-self.wlen]
        self.stds_ = np.sqrt(stds  - self.means_**2)

    def first_line(self,i): 
        """Compute the line of the crossdistance matrix at index i

        Args:
            i (int): index where start the computation of the chunk

        Returns:
            np.ndarray: crossditance matrix line at index i
        """

        self.first_idx_ = i
        self.idx_ = i

        if i!=0:
            self.dot_product_ = np.convolve(self.signal_[i:i+self.wlen][::-1],self.signal_,'valid')
        else: 
            self.dot_product_ = self.first_dot_product_.copy()

        dist = (self.dot_product_ - self.wlen*self.means_[i]*self.means_)/(self.wlen*self.stds_[i]*self.stds_)

        return np.sqrt(np.clip(2*self.wlen*(1-dist),0,None))

    def individual_distance(self,i,j): 
        a = (self.signal_[i:i+self.wlen]-self.means_[i])/self.stds_[i]
        b = (self.signal_[j:j+self.wlen]-self.means_[j])/self.stds_[j]
        return np.sqrt(np.sum((a-b)**2))


    def next_line(self)->np.ndarray:
        """Iterator to compute the line of the crossdistance matrix

        Returns:
            np.ndarray: crossdistance matrix line
        """
        self.dot_product_[1:] = self.dot_product_[:-1] + \
            self.signal_[self.idx_+self.wlen]*self.signal_[self.wlen:] - \
                self.signal_[self.idx_]*self.signal_[:-self.wlen]
        self.idx_ +=1 
        self.dot_product_[0] = self.first_dot_product_[self.idx_]
        dist = (self.dot_product_ - self.wlen*self.means_[self.idx_]*self.means_)/(self.wlen*self.stds_[self.idx_]*self.stds_)
        dist = np.sqrt(np.clip(2*self.wlen*(1-dist),0,None))
        return dist    
    
#########################################################################################################################################
#########################################################################################################################################

class UnitEuclidean(object): #Verifed

    def __init__(self,wlen:int,**kwargs)->None:
        """Initialization

        Args:
            wlen (int): window length
        """
        self.wlen = wlen

    def fit(self,signal:np.ndarray):
        """Compute the elementary components of the first line of the crossdistance matrix. 

        Args:
            signal (np.ndarray): signal. shape (n_samples,n_dimension)
        """
        self.signal_ =  signal
        self.first_basic_ = self._first_elemntary()

    def _first_elemntary(self)->np.ndarray:
        """Compute the dotproduct between subsequence of the first line of the crossdistance matrix. 
        Compute the mean and std for each subsequence. 
        """
        self.first_dot_product_ = np.convolve(self.signal_[:self.wlen][::-1],self.signal_,'valid')

        means = np.zeros(len(self.signal_)+1)
        means[1:] = np.cumsum(self.signal_)/self.wlen
        self.means_ = means[self.wlen:]-means[:-self.wlen]

        stds = np.zeros(len(self.signal_)+1)
        stds[1:] =np.cumsum(self.signal_**2)/self.wlen
        stds = stds[self.wlen:]-stds[:-self.wlen]
        self.stds_ = np.sqrt(stds  - self.means_**2)

    def first_line(self,i): 
        """Compute the line of the crossdistance matrix at index i

        Args:
            i (int): index where start the computation of the chunk

        Returns:
            np.ndarray: crossditance matrix line at index i
        """

        self.first_idx_ = i
        self.idx_ = i

        if i!=0:
            self.dot_product_ = np.convolve(self.signal_[i:i+self.wlen][::-1],self.signal_,'valid')
        else: 
            self.dot_product_ = self.first_dot_product_.copy()

        dist = (self.dot_product_ - self.wlen*self.means_[i]*self.means_)/(self.wlen*self.stds_[i]*self.stds_)

        return np.sqrt(np.clip(2*(1-dist),0,None))

    def individual_distance(self,i,j): 
        a = (self.signal_[i:i+self.wlen]-self.means_[i])/self.stds_[i]
        b = (self.signal_[j:j+self.wlen]-self.means_[j])/self.stds_[j]
        return np.sqrt(np.sum((a-b)**2))


    def next_line(self)->np.ndarray:
        """Iterator to compute the line of the crossdistance matrix

        Returns:
            np.ndarray: crossdistance matrix line
        """
        self.dot_product_[1:] = self.dot_product_[:-1] + \
            self.signal_[self.idx_+self.wlen]*self.signal_[self.wlen:] - \
                self.signal_[self.idx_]*self.signal_[:-self.wlen]
        self.idx_ +=1 
        self.dot_product_[0] = self.first_dot_product_[self.idx_]
        dist = (self.dot_product_ - self.wlen*self.means_[self.idx_]*self.means_)/(self.wlen*self.stds_[self.idx_]*self.stds_)
        dist = np.sqrt(np.clip(2*(1-dist),0,None))
        return dist    
    
#########################################################################################################################################
#########################################################################################################################################

class LTNormalizedEuclidean(object): 

    def __init__(self,wlen:int,**kwargs)->None:
        """Initialization

        Args:
            wlen (int): window length
        """
        self.wlen = wlen

    def fit(self,signal:np.ndarray):
        """Compute the elementary components of the first line of the crossdistance matrix. 

        Args:
            signal (np.ndarray): signal. shape (n_samples,n_dimension)
        """
        self.signal_ =  signal
        self.first_basic_ = self._first_elemntary()

    def _first_elemntary(self)->np.ndarray:
        """Compute the dotproduct between subsequence of the first line of the crossdistance matrix. 
        Compute the mean and std for each subsequence. 
        """
        self.first_dot_product_ = np.convolve(self.signal_[:self.wlen][::-1],self.signal_,'valid')

        means = np.zeros(len(self.signal_)+1)
        means[1:] = np.cumsum(self.signal_)/self.wlen
        self.means_ = means[self.wlen:]-means[:-self.wlen]

        stds = np.zeros(len(self.signal_)+1)
        stds[1:] =np.cumsum(self.signal_**2)/self.wlen
        stds = stds[self.wlen:]-stds[:-self.wlen]
        self.stds_ = np.sqrt(stds  - self.means_**2)

        self.stdt = np.sqrt((self.wlen**2 -1)/12)

        self.alphas_ = np.convolve(np.arange(self.wlen)[::-1],self.signal_,'valid')/self.wlen - self.means_*(self.wlen-1)/2
        self.alphas_ = self.alphas_/self.stdt**2
        self.etas_ = np.sqrt(self.wlen*(self.stds_**2 - self.alphas_**2 * self.stdt**2))

        

    def first_line(self,i): 
        """Compute the line of the crossdistance matrix at index i

        Args:
            i (int): index where start the computation of the chunk

        Returns:
            np.ndarray: crossditance matrix line at index i
        """

        self.first_idx_ = i
        self.idx_ = i

        if i!=0:
            self.dot_product_ = np.convolve(self.signal_[i:i+self.wlen][::-1],self.signal_,'valid')
        else: 
            self.dot_product_ = self.first_dot_product_.copy()

        dist = (self.dot_product_ - self.wlen*(self.means_[i]*self.means_+self.stdt**2 * self.alphas_[i]*self.alphas_))/(self.etas_[i]*self.etas_)

        return np.sqrt(np.clip(2*(1-dist),0,None))


    def next_line(self)->np.ndarray:
        """Iterator to compute the line of the crossdistance matrix

        Returns:
            np.ndarray: crossdistance matrix line
        """
        self.dot_product_[1:] = self.dot_product_[:-1] + \
            self.signal_[self.idx_+self.wlen]*self.signal_[self.wlen:] - \
                self.signal_[self.idx_]*self.signal_[:-self.wlen]
        self.idx_ +=1 
        self.dot_product_[0] = self.first_dot_product_[self.idx_]
        dist = (self.dot_product_ - self.wlen*(self.means_[self.idx_]*self.means_+ self.stdt**2 * self.alphas_[self.idx_]*self.alphas_))/(self.etas_[self.idx_]*self.etas_)
        dist = np.sqrt(np.clip(2*(1-dist),0,None))
        return dist    