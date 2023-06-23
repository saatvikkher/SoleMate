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
    def __init__(self, pair: SolePair, downsample_rate=0.2, random_seed=47) -> None:
        '''
        Inputs:
            Q: (Pandas DataFrame) for shoe Q
            K: (Pandas DataFrame) for shoe K
        '''
        self.Q_coords, self.K_coords = pair.icp_transform()
        self.Q_coords = self.Q_coords.sample(frac=downsample_rate, random_state=random_seed)
        self.K_coords = self.K_coords.sample(frac=downsample_rate, random_state=random_seed)
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
    
    def min_dist(self, Q_as_base=True, sample_num=500, random_seed=1):
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
            sample_num (int): number of points sampled from base dataframe
            random_seed (int)
        
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
        query_points = df2.sample(n=sample_num, random_state=random_seed)[['x', 'y']].values
        
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


if __name__ == "__main__":
    t0 = time.time()
    Q = Sole("2DScanImages/001351R_20171031_2_1_1_csafe_bpbryson.tiff")
    K = Sole("2DScanImages/001351R_20171031_2_1_2_csafe_bpbryson.tiff")
    t1 = time.time()
    # print("Time for 2 Soles(): " + str(t1-t0))
    Q.plot()
    K.plot()

    t0 = time.time()
    pair = SolePair(Q, K, mated=True)
    t1 = time.time()
    # print("Time for SolePair(): " + str(t1-t0))
    pair.plot()
    
    t0 = time.time()
    sc = SolePairCompare(pair) # icp is called here
    t1 = time.time()
    print("Time for SolePairCompare(): " + str(t1-t0))
    pair.plot(aligned=True)

    t0 = time.time()
    print(sc.percent_overlap(Q_as_base=True, threshold=4))
    print(sc.percent_overlap(Q_as_base=False, threshold=4))
    t1 = time.time()
    # print("Time for 2 overlaps: " + str(t1-t0))

    t0 = time.time()
    print(sc.min_dist())
    t1 = time.time()
    # print("Time for min_distance: " + str(t1-t0))

    # t0 = time.time()
    # print(sc.min_dist(sample_num=3000))
    # t1 = time.time()
    # print(t1-t0)

    # t0 = time.time()
    # print(sc.min_dist(sample_num=len(sc.Q_coords)))
    # t1 = time.time()
    # print(t1-t0)

    # # Define the range of the changing number
    # x_values = [0,1,2,3,4,5,6,7,8,9,10]

    # # Initialize empty lists to store the percent overlap values
    # percent_overlap_q_km = []
    # percent_overlap_k_qm = []
    # percent_overlap_q_knm = []
    # percent_overlap_k_qnm = []

    # # Calculate the percent overlap for each value of x
    # for x in x_values:
    #     percent_overlap_q_km.append(sc.percent_overlap(q_km, k, x))
    #     percent_overlap_k_qm.append(sc.percent_overlap(k, q_km, x))
    #     percent_overlap_q_knm.append(sc.percent_overlap(q_knm, k, x))
    #     percent_overlap_k_qnm.append(sc.percent_overlap(k, q_knm, x))

    # # Plot the curves
    # plt.plot(x_values, percent_overlap_q_km, label="q_km vs k")
    # plt.plot(x_values, percent_overlap_k_qm, label="k vs q_km")
    # plt.plot(x_values, percent_overlap_q_knm, label="q_knm vs k")
    # plt.plot(x_values, percent_overlap_k_qnm, label="k vs q_knm")

    # # Add labels and legend
    # plt.xlabel("Changing Number (x)")
    # plt.ylabel("Percent Overlap")
    # plt.legend()

    # # Show the plot
    # plt.show()

    # print("q_km vs k")
    # print(percent_overlap_q_km)
    # print()

    # print("k vs q_km")
    # print(percent_overlap_k_qm)
    # print()

    # print("q_knm vs k")
    # print(percent_overlap_q_knm)
    # print()

    # print("k vs q_knm")
    # print(percent_overlap_k_qnm)


# q_km vs k
# [0.13938494893592743, 0.5350887202601701, 0.7316568722542364, 0.8560078735664974, 0.9132766588691733, 0.946767843897986, 0.9639270839276545, 0.9724139898442403, 0.9774918696867689, 0.9813715981057797, 0.9838106920750842]

# k vs q_km
# [0.12776528424245595, 0.49428638669525654, 0.6821295957324408, 0.8048088489095758, 0.8659719679933058, 0.8991553789027771, 0.9194995031640605, 0.9308090581036557, 0.9381962240468594, 0.943020762512421, 0.9462632707494378]

# q_knm vs k
# [0.07614495478438174, 0.30698955340474265, 0.481907132548625, 0.622174851377653, 0.7219007772055027, 0.7863810293007798, 0.8328412355846254, 0.8646311914944325, 0.8878281938909264, 0.9064837740145908, 0.9207170945489692]

# k vs q_knm
# [0.0751791224308352, 0.2973955337063961, 0.4596647664871084, 0.5922938130850897, 0.694798912190785, 0.7736127817582762, 0.8288661680874432, 0.8647298781444485, 0.8894801527116782, 0.9088175304638879, 0.9244155640395377]