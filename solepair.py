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
    
    
    def icp_transform(self, max_iterations: int = 10000, 
                      tolerance: float = 0.00001, 
                      downsample_rate: float = 1.0, 
                      random_seed: int = 47, 
                      shift_left=False, 
                      shift_right=False, 
                      shift_up=False, 
                      shift_down=False) -> pd.DataFrame | pd.DataFrame:
        '''
        Default: do 1 icp with no initial shift
        The user can specify up to four additional icp trials with differnt initial shifts
        '''
        print("Q coords before: ", self.Q.coords)

        # Default no shift
        shifts = [(0, 0)]
        

        range = 2 * max((self.Q.coords.x.max() - self.Q.coords.x.min()), (self.Q.coords.y.max() - self.Q.coords.y.min()))
        print("range: ", range)
        
        if shift_left:
            shifts.append((-range, 0))
        if shift_right:
            shifts.append((range, 0))
        if shift_down:
            shifts.append((0, -range))
        if shift_up:
            shifts.append((0, range))
        
        print("SHIFT: ", shifts)

        best_T = None
        best_rmse = np.Inf 
        best_shift = None

        for shift in shifts:
            # apply shift
            self.Q.coords.loc[:,"x"] += shift[0]
            self.Q.coords.loc[:,"y"] += shift[1]
            
            T, rmse = self._icp_helper(max_iterations=max_iterations, tolerance=tolerance, downsample_rate=downsample_rate, random_seed=random_seed)
            print("RMSE calculation: ", rmse)
            if rmse < best_rmse:
                best_rmse = rmse
                best_T = T
                best_shift = shift
                print("Best RMSE: ", best_rmse)

            # reverse shift
            self.Q.coords.loc[:,"x"] -= shift[0]
            self.Q.coords.loc[:,"y"] -= shift[1]
        
        self.T = best_T

        print("Q coords: ", self.Q.coords)

        # Apply the best_shift
        self.Q.coords.loc[:,"x"] += best_shift[0]
        self.Q.coords.loc[:,"y"] += best_shift[1]
        
        transformed_q = pd.DataFrame(self._transform(self.Q.coords, self.T))
        self.Q.aligned_coordinates = transformed_q.rename(columns = {0: 'x', 1 : 'y'})
        self.aligned = True

        # reverse best_shift
        self.Q.coords.loc[:,"x"] -= best_shift[0]
        self.Q.coords.loc[:,"y"] -= best_shift[1]

        return self.Q.aligned_coordinates, self.K.coords


    def _icp_helper(self, max_iterations: int = 10000, 
                      tolerance: float = 0.00001, 
                      downsample_rate: float = 1.0, 
                      random_seed: int = 47):
        
        q_pts, k_pts = self._equalize()
        np.random.seed(random_seed)
        num_samples = int(q_pts.shape[0] * downsample_rate)
        sample_indices_q = np.random.choice(q_pts.shape[0], num_samples, replace=False)
        sample_indices_k = np.random.choice(k_pts.shape[0], num_samples, replace=False)
        q_pts = q_pts[sample_indices_q]
        k_pts = k_pts[sample_indices_k]

        T, distances, _ = icp(q_pts,k_pts, max_iterations=max_iterations, tolerance=tolerance)

        # RMSE (Root Mean Squared Error) calc
        rmse = np.sqrt(np.mean(distances)**2)

        return T, rmse
    
    def plot(self, size: float = 0.5, aligned: bool = False):
        '''
        Inputs:
            aligned: (bool) indicating if you want to access the aligned image
        '''

        plt.scatter(self.K.coords.x, self.K.coords.y, s = size, label = "K", color = "#500082")

        if aligned:
            plt.scatter(self.Q.aligned_coordinates.x, self.Q.aligned_coordinates.y, s=size, label="Aligned Q", color="#FFBE0A")
        else:
            plt.scatter(self.Q.coords.x, self.Q.coords.y, s=size, label="Q", color="#FFBE0A")

        plt.legend()

        plt.show()

