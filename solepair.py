import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sole import Sole
from icp import icp, nearest_neighbor
from util import WILLIAMS_PURPLE, WILLIAMS_GOLD


class SolePair():
    def __init__(self, Q: Sole, K: Sole, mated: bool, aligned=False, T: np.ndarray = np.identity(3, int)) -> None:
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
        else:
            return q_pts, k_pts

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
                      shift_down=False,
                      two_way=False,
                      overlap_threshold=3) -> pd.DataFrame | pd.DataFrame:
        '''
        Default: do 1 icp with no initial shift
        The user can specify up to four additional icp trials with differnt initial shifts
        '''

        # Default: no shift
        shifts = [(0, 0)]

        range = 2 * max((self.Q.coords.x.max() - self.Q.coords.x.min()),
                        (self.Q.coords.y.max() - self.Q.coords.y.min()))

        if shift_left:
            shifts.append((-range, 0))
        if shift_right:
            shifts.append((range, 0))
        if shift_down:
            shifts.append((0, -range))
        if shift_up:
            shifts.append((0, range))

        best_T = None
        best_percent_overlap = -1
        best_shift = None
        # record apply_to_q for the transformation that got the best percent overlap
        best_apply_to_q = None

        for shift in shifts:
            # apply shift
            self.Q.coords.loc[:, "x"] += shift[0]
            self.Q.coords.loc[:, "y"] += shift[1]

            T, percent_overlap, apply_to_q = self._icp_helper(max_iterations=max_iterations,
                                                              tolerance=tolerance,
                                                              downsample_rate=downsample_rate,
                                                              random_seed=random_seed,
                                                              two_way=two_way,
                                                              overlap_threshold=overlap_threshold)

            # Percent overlap: higher the better
            if percent_overlap > best_percent_overlap:
                best_percent_overlap = percent_overlap
                best_T = T
                best_shift = shift
                best_apply_to_q = apply_to_q

            # reverse shift
            self.Q.coords.loc[:, "x"] -= shift[0]
            self.Q.coords.loc[:, "y"] -= shift[1]

        # If we are not applying the transformation matrix to Q,
        # we get the reversed T first and apply it to Q later.
        # Note: the reverse of a homogenous transformation is its inverse
        if not best_apply_to_q:
            best_T = np.linalg.inv(best_T)

        self.T = best_T

        # Apply the best_shift
        self.Q.coords.loc[:, "x"] += best_shift[0]
        self.Q.coords.loc[:, "y"] += best_shift[1]

        transformed_q = pd.DataFrame(
            self._transform(self.Q.coords.to_numpy(), self.T))

        self.Q.aligned_coordinates = transformed_q.rename(
            columns={0: 'x', 1: 'y'})

        self.aligned = True

        # reverse best_shift
        self.Q.coords.loc[:, "x"] -= best_shift[0]
        self.Q.coords.loc[:, "y"] -= best_shift[1]

        return self.Q.aligned_coordinates, self.K.coords

    def _icp_helper(self, max_iterations: int = 10000,
                    tolerance: float = 0.00001,
                    downsample_rate: float = 1.0,
                    random_seed: int = 47,
                    two_way: bool = False,
                    overlap_threshold=3):

        q_pts, k_pts = self._equalize()
        np.random.seed(random_seed)
        num_samples = int(q_pts.shape[0] * downsample_rate)
        sample_indices_q = np.random.choice(
            q_pts.shape[0], num_samples, replace=False)
        sample_indices_k = np.random.choice(
            k_pts.shape[0], num_samples, replace=False)
        q_pts = q_pts[sample_indices_q]
        k_pts = k_pts[sample_indices_k]

        T_qk, distances_qk, _ = icp(q_pts, k_pts,
                                    max_iterations=max_iterations,
                                    tolerance=tolerance)

        # Calculate percent overlap
        percent_overlap_qk = np.sum(
            distances_qk <= overlap_threshold) / num_samples

        apply_to_q = True

        if two_way:
            T_kq, _, __ = icp(k_pts, q_pts,
                              max_iterations=max_iterations,
                              tolerance=tolerance)

            distances_kq_q_as_base, _ = nearest_neighbor(
                q_pts, self._transform(k_pts, T_kq))

            # Now, we have the distances_kq_q_as_base to calculate percent_overlap_kq
            percent_overlap_kq = np.sum(
                distances_kq_q_as_base <= overlap_threshold) / num_samples

            # percent_overlap -- higher the better.
            if percent_overlap_kq < percent_overlap_qk:
                apply_to_q = False
                return T_kq, percent_overlap_kq, apply_to_q

        return T_qk, percent_overlap_qk, apply_to_q

    def plot(self,
             size: float = 0.1,
             aligned: bool = False,
             color_q=WILLIAMS_GOLD,
             color_k=WILLIAMS_PURPLE):
        '''
        Inputs:
            aligned: (bool) indicating if you want to access the aligned image
        '''

        plt.scatter(self.K.coords.x, self.K.coords.y,
                    s=size, label="K", color=color_k)

        if aligned:
            plt.scatter(self.Q.aligned_coordinates.x, self.Q.aligned_coordinates.y,
                        s=size, label="Aligned Q", color=color_q)
        else:
            plt.scatter(self.Q.coords.x, self.Q.coords.y,
                        s=size, label="Q", color=color_q)

        plt.legend()

        plt.show()
