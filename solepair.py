import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from sole import Sole
from icp import icp

class SolePair():
    def __init__(self, Q: Sole, K: Sole, mated: bool, aligned = False, T: np.ndarray= np.identity(3, int)) -> None:
        self._Q = Q 
        self._K = K 
        self._mated = mated
        self._aligned = aligned
        self._T = T

    @property
    def Q(self) -> Sole:
        return self._Q

    @property
    def K(self) -> Sole:
        return self._K

    @property
    def mated(self) -> bool:
        return self._mated

    @property
    def aligned(self) -> bool:
        return self._aligned
    
    @property
    def T(self) -> bool:
        return self._T

    @aligned.setter
    def aligned(self, value: bool) -> None:
        self._aligned = value

    @T.setter
    def T(self, value: np.ndarray) -> None:
        self._T = value

    def _equalize(self) -> np.ndarray | np.ndarray:
        q_pts = np.array(self.Q.coords)
        k_pts = np.array(self.K.coords)
        
        if q_pts.shape[0] > k_pts.shape[0]:
            return q_pts[np.random.choice(q_pts.shape[0], k_pts.shape[0], replace=False), :], k_pts
        elif q_pts.shape[0] < k_pts.shape[0]:
            return q_pts, k_pts[np.random.choice(k_pts.shape[0], q_pts.shape[0], replace=False), :]
        else: return q_pts, k_pts

    def _transform(self, q_pts: np.ndarray, T: np.ndarray) -> np.ndarray:
        q_hom = np.hstack((q_pts, np.ones((q_pts.shape[0], 1))))
        t_q_hom = np.dot(q_hom, T.T)
        transformed_q = t_q_hom[:, :2] / t_q_hom[:, 2][:, np.newaxis]
    
        return transformed_q
    
    
    def icp_transform(self, max_iterations: int = 10000, tolerance: float = 0.00001, downsample_rate: float = 1.0, random_seed: int = 47) -> pd.DataFrame | pd.DataFrame:
        q_pts, k_pts = self._equalize()

        num_samples = int(q_pts.shape[0] * downsample_rate)
        sample_indices_q = np.random.choice(q_pts.shape[0], num_samples, replace=False)
        sample_indices_k = np.random.choice(k_pts.shape[0], num_samples, replace=False)
        q_pts = q_pts[sample_indices_q]
        k_pts = k_pts[sample_indices_k]

        self.T,_,i = icp(q_pts,k_pts, max_iterations=max_iterations, tolerance=tolerance)
        transformed_q = pd.DataFrame(self._transform(self.Q.coords, self.T))
        self.Q.aligned_coordinates = transformed_q.rename(columns = {0: 'x', 1 : 'y'})
        self.aligned = True
        return self.Q.aligned_coordinates, self.K.coords
    
    def plot(self, size: float = 0.5, aligned: bool = False):
        '''
        Inputs:
            aligned: (bool) indicating if you want to access the aligned image
        '''

        if aligned:
            plt.scatter(self.Q.aligned_coordinates.x, self.Q.aligned_coordinates.y, s = size, label="Aligned Q")
        else:
            plt.scatter(self.Q.coords.x, self.Q.coords.y, s = size, label="Q")
        
        plt.scatter(self.K.coords.x, self.K.coords.y, s = size, label = "K")

        plt.legend()

        plt.show()

