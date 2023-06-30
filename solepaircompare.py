import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.spatial import KDTree
from scipy.stats import kurtosis
import math
import time
from solepair import SolePair
from sole import Sole


class SolePairCompare:
    '''
    SolePairCompare will take two shoeprint images (Q and K) and eventually 
    outputs the collection of similarity metrics (including ...) between Q and K.

    This class also provides methods for intermediate steps in this pipeline.
    '''

    def __init__(self, pair: SolePair, downsample_rate=0.2, icp_downsample_rate=1.0, random_seed=47,
                 shift_left=False,
                 shift_right=False,
                 shift_up=False,
                 shift_down=False) -> None:
        '''
        Inputs:
            Q: (Pandas DataFrame) for shoe Q
            K: (Pandas DataFrame) for shoe K
        '''
        self.Q_coords, self.K_coords = pair.icp_transform(
            downsample_rate=icp_downsample_rate, shift_left=shift_left, shift_right=shift_right, shift_up=shift_up, shift_down=shift_down)
        self.Q_coords = self.Q_coords.sample(
            frac=downsample_rate, random_state=random_seed)
        self.K_coords = self.K_coords.sample(
            frac=downsample_rate, random_state=random_seed)
        self.pair = pair

    def _df_to_hash(self, df):
        '''
        Changes a dataframe to a hashtable.

        Inputs: df (dataframe)

        Returns: (defaultdict)
        '''
        hashtable = defaultdict(set)
        for _, row in df.iterrows():
            # We assume the column names are "x" and "y"
            hashtable[row['x']].add(row['y'])
        return hashtable

    def _is_overlap(self, x, y, ht, threshold):
        '''
        Determines whether the point given by (x, y) has any overlapping points 
        with the hashtable. We define overlap as being within the given 
        threshold. We look for points around (x, y) according the threshold 
        value. There is overlap if there exists at least one point from ht such 
        that it is in the circular region with radius of threshold
        around (x, y).

        Inputs: 
            x (int)
            y (int)
            ht (defaultdict)
            threshold (int)

        Returns: (bool)
        '''
        for potential_x in range(x-threshold, x+threshold+1):
            for potential_y in range(y-threshold, y+threshold+1):
                if potential_x in ht and potential_y in ht[potential_x]:
                    # also check whether that matched point is in circle of radius = threshold
                    if math.dist([x, y], [potential_x, potential_y]) <= threshold:
                        return True
        return False

    def percent_overlap(self, Q_as_base=True, threshold=3):
        '''
        (Similarity metric) 

        Calculates the percent of points in the base shoeprint (determined by
        Q_as_base) that overlap with points in the other shoeprint.

        Inputs: 
            Q_as_base (bool): if Q is the base dataframe, defaults to True
            threshold (int): if distance between two points <= threshold, 
                             it qualifies as an overlap. 

        Returns: proportion of points that overlap (float)
        '''

        # Round Q and K coords because the hashtable implementation in
        # _df_to_hash relies on integer values of coordinates
        if Q_as_base:
            df1 = self.Q_coords.round().astype(int)
            df2 = self.K_coords.round().astype(int)
        else:
            df1 = self.K_coords.round().astype(int)
            df2 = self.Q_coords.round().astype(int)

        ht = self._df_to_hash(df2)
        overlap_count = df1.apply(lambda point: self._is_overlap(point['x'],
                                                                 point['y'], ht, threshold), axis=1).sum()
        return overlap_count/len(df1)

    def min_dist(self, Q_as_base=True):
        '''
        (Similarity metric) 

        min_dist is a similarity metric comparing the dataframes of K and Q. 
        For each sample point from the base dataframe, min_dist finds the Euclidean 
        distance of the closest point in the other dataframe using the 
        kdtree closest neighbor algorithm. 

        It outputs distribution statistics including the average, 
        standard deviation, and 0.1, 0.25, 0.5, 0.75, 0.9 quantile values, as well as
        kurtosis.

        Inputs:
            Q_as_base (bool): if Q is the base dataframe, defaults to True

        Returns:
            (dict): a dictionary of statistics with keys "mean", "std", "0.1", 
                    "0.25", "0.5", "0.75", "0.9", "kurtosis"
        '''

        # If Q_as_base == True, Q is the shoe we sample from
        if Q_as_base:
            df1 = self.K_coords
            df2 = self.Q_coords
        else:
            df1 = self.Q_coords
            df2 = self.K_coords

        min_dists = []

        # We assume the column names are "x" and "y"
        points = df1[["x", "y"]].values
        kdtree = KDTree(points)
        query_points = df2[['x', 'y']].values

        for query_point in query_points:
            dist, _ = kdtree.query(query_point)
            min_dists.append(dist)

        min_dists_arr = np.array(min_dists)
        min_dists_dict = {}
        min_dists_dict["mean"] = np.mean(min_dists_arr)
        min_dists_dict["std"] = np.std(min_dists_arr)
        min_dists_dict["0.1"] = np.quantile(min_dists_arr, 0.1)
        min_dists_dict["0.25"] = np.quantile(min_dists_arr, 0.25)
        min_dists_dict["0.5"] = np.quantile(min_dists_arr, 0.5)
        min_dists_dict["0.75"] = np.quantile(min_dists_arr, 0.75)
        min_dists_dict["0.9"] = np.quantile(min_dists_arr, 0.9)
        min_dists_dict["kurtosis"] = kurtosis(min_dists_arr)

        return min_dists_dict
