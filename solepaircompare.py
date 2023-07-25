import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.spatial import KDTree
from scipy.stats import kurtosis
import math
from solepair import SolePair
from sklearn.cluster import AgglomerativeClustering, KMeans
from skimage.metrics import structural_similarity as ssim

class SolePairCompare:
    '''
    SolePairCompare will take two shoeprint images (Q and K) and eventually 
    outputs the collection of similarity metrics (including ...) between Q and K.

    This class also provides methods for intermediate steps in this pipeline.
    '''

    def __init__(self,
                 pair: SolePair,
                 downsample_rate=1.0,
                 icp_downsample_rates=[1.0],
                 random_seed=0,
                 shift_left=False,
                 shift_right=False,
                 shift_up=False,
                 shift_down=False,
                 two_way=False,
                 icp_overlap_threshold=3) -> None:
        '''
        Inputs:
            Q: (Pandas DataFrame) for shoe Q
            K: (Pandas DataFrame) for shoe K
        '''

        # Sorting the icp_downsample_rates to optimize time efficiency of the short circuit
        icp_downsample_rates.sort()
            
        best_icp_downsample_rate = None
        best_percent_overlap = -1

        for icp_downsample_rate in icp_downsample_rates:
            # pair.icp_transform sets self._aligned, self._T, and self.K.aligned_coordinates in place
            self._Q_coords, self._K_coords, self._Q_coords_full, self._K_coords_full = pair.icp_transform(downsample_rate=icp_downsample_rate,
                                                                                                         shift_left=shift_left,
                                                                                                         shift_right=shift_right,
                                                                                                         shift_up=shift_up,
                                                                                                         shift_down=shift_down,
                                                                                                         two_way=two_way,
                                                                                                         overlap_threshold=icp_overlap_threshold)

            self._Q_coords = self._Q_coords.sample(frac=downsample_rate, random_state=random_seed)
            self._K_coords = self._K_coords.sample(frac=downsample_rate, random_state=random_seed)
            
            # Check the percent overlap of this icp_downsample_rate
            po = self.percent_overlap()

            # higher the better
            if po > best_percent_overlap:
                best_percent_overlap = po
                best_icp_downsample_rate = icp_downsample_rate

        # Short circuit: if the best_icp_downsample_rate equals the last
        # element in icp_downsample_rates, then we don't need to run the ICP 
        # again to update pair.T in place.
        if icp_downsample_rates[-1] != best_icp_downsample_rate:
            self._Q_coords, self._K_coords, self._Q_coords_full, self._K_coords_full = pair.icp_transform(downsample_rate=best_icp_downsample_rate,
                                                                                                        shift_left=shift_left,
                                                                                                        shift_right=shift_right,
                                                                                                        shift_up=shift_up,
                                                                                                        shift_down=shift_down,
                                                                                                        two_way=two_way,
                                                                                                        overlap_threshold=icp_overlap_threshold)

            self._Q_coords = self._Q_coords.sample(frac=downsample_rate, random_state=random_seed)
            self._K_coords = self._K_coords.sample(frac=downsample_rate, random_state=random_seed)

        # The percent of original K that is keeping
        self.K_keep_percent = 1.0
        self.pair = pair
        self.random_seed = random_seed
    
    @property
    def Q_coords(self) -> pd.DataFrame:
        return self._Q_coords

    @property
    def K_coords(self) -> pd.DataFrame:
        return self._K_coords

    @property
    def Q_coords_full(self) -> pd.DataFrame:
        return self._Q_coords_full

    @property
    def K_coords_full(self) -> pd.DataFrame:
        return self._K_coords_full
    
    @Q_coords.setter
    def Q_coords(self, value) -> None:
        '''Setter method for Q dataframe of coordinates'''
        self._Q_coords = value
    
    @K_coords.setter
    def K_coords(self, value) -> None:
        '''Setter method for K dataframe of coordinates'''
        self._K_coords = value

    @Q_coords.setter
    def Q_coords_full(self, value) -> None:
        '''Setter method for Q dataframe of coordinates'''
        self._Q_coords_full = value
    
    @K_coords.setter
    def K_coords_full(self, value) -> None:
        '''Setter method for K dataframe of coordinates'''
        self._K_coords_full = value

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
        def get_dist(point_a, point_b):
            x1, y1 = point_a[0], point_a[1]
            x2, y2 = point_b[0], point_b[1]
            return math.hypot(x1-x2, y1-y2)

        for potential_x in range(x-threshold, x+threshold+1):
            for potential_y in range(y-threshold, y+threshold+1):
                if potential_x in ht and potential_y in ht[potential_x]:
                    # also check whether that matched point is in circle of radius = threshold
                    if get_dist([x, y], [potential_x, potential_y]) <= threshold:
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
        overlap_count = df1.apply(lambda point: self._is_overlap(
            point['x'], point['y'], ht, threshold), axis=1).sum()
        return overlap_count/len(df1)

    def min_dist(self, Q_as_base=True):
        '''
        (Similarity metric) 

        min_dist is a similarity metric comparing the dataframes of K and Q. 
        For each sample point from the base dataframe, min_dist finds the Euclidean 
        distance of the closest point in the other dataframe using the 
        kdtree closest neighbor algorithm. 

        It outputs distribution statistics including the average, 
        standard deviation, and 0.1, 0.25, 0.5, 0.75, 0.9 quantile values.

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
        # min_dists_dict["kurtosis"] = kurtosis(min_dists_arr)

        return min_dists_dict

    def _hierarchical_cluster(self, df: pd.DataFrame, n_clusters: int) -> pd.DataFrame:
        '''
        This function takes in pandas DataFrame and runs hierarchical clustering
        on them. It returns the centroid points of the hierarchical clustering.

        Inputs:
            data (np.array, shape of nx2)
            n_clusters (int): the number of clusters generated by h. clustering

        Returns:
            centroid_pts (pd.dataframe with columns 'x' and 'y')
        '''
        df_arr = df.to_numpy()
        df_label = df.copy(deep=True)
        hierarchical_cluster = AgglomerativeClustering(n_clusters=n_clusters,
                                                       affinity='euclidean')
        df_label['label'] = hierarchical_cluster.fit_predict(df_arr)
        centroids = pd.DataFrame(columns=['x', 'y'])
        for i in range(n_clusters):
            centroids.loc[len(centroids)] = [df_label[df_label["label"] == i]["x"].mean(),
                                             df_label[df_label["label"] == i]["y"].mean()]
        return centroids

    def _kmeans_cluster(self, df: pd.DataFrame, init: pd.DataFrame, n_clusters: int):
        '''
        Runs kmeans clustering, with n_clusters amount of clusters and given 
        initial points. 

        Inputs:
            df: x, y coordinates of a shoe
            init: the points where kmeans clustering begins
            n_clusters: the number of clusters

        Returns:
            centroids (pd.DataFrame)
            df_labels (pd.DataFrame with columns "x", "y", and "label")
            kmeans (KMeans object)
        '''
        assert init.shape[0] == n_clusters

        df_arr = df.to_numpy()
        df_labels = df.copy(deep=True)
        kmeans = KMeans(n_clusters=n_clusters, init=init, n_init=1)
        kmeans.fit(df_arr)

        df_labels['label'] = kmeans.labels_
        _ = kmeans.cluster_centers_
        centroids = pd.DataFrame(_, columns=['x', 'y'])

        return centroids, df_labels, kmeans

    def _centroid_distance_metric(self, centroids_a: pd.DataFrame, centroids_b: pd.DataFrame) -> float:
        '''
        Computes the centroid distance metric. Given two sets of centroids, 
        compute the distance between each paired centroid. Then, compute the 
        rmse of these distances. Since rmse is always postive, centroids_a can 
        be either shoe Q or shoe K.

        Inputs:
            centroids_a: dataframe of n_clusters points, has columns "x" and "y"
            centorids_b: dataframe of n_clusters points, has columns "x" and "y"

        Returns:
            rmse (float)
        '''
        assert centroids_a.shape == centroids_b.shape

        centroids_a = centroids_a.to_numpy()
        centroids_b = centroids_b.to_numpy()

        centr_differences = centroids_a - centroids_b
        distances_squared = np.sum(centr_differences ** 2, axis=1)
        mean_squared_distance = np.mean(distances_squared)
        rmse = math.sqrt(mean_squared_distance)

        return rmse

    def _cluster_prop_metric(self, df_a: pd.DataFrame, df_b: pd.DataFrame, n_clusters: int) -> float:
        '''
        TODO: 

        S I M O N
        I I D O C
        M D M ? ?
        O O ? O !
        N C ? ! N
        '''
        cluster_prop_diff = []
        len_a = df_a.shape[0]
        len_b = df_b.shape[0]

        for i in range(n_clusters):
            num_in_cluster_a = df_a[df_a['label'] == i].shape[0]
            num_in_cluster_b = df_b[df_b['label'] == i].shape[0]
            cluster_prop_diff.append(
                num_in_cluster_a / len_a - num_in_cluster_b / len_b)

        cluster_prop_diff_arr = np.array(cluster_prop_diff)
        squared_diff = cluster_prop_diff_arr ** 2
        mean_squared_diff = np.mean(squared_diff)
        rmse = np.sqrt(mean_squared_diff)

        return rmse

    def _cluster_var(self, df: pd.DataFrame, centroid: pd.DataFrame) -> float:
        df_arr = df.to_numpy()
        distance = np.linalg.norm(df_arr - centroid, axis=1)
        distance_squared = distance ** 2
        sum_distance_squared = np.sum(distance_squared)
        return sum_distance_squared

    def _within_cluster_var(self, df: pd.DataFrame, centroids: pd.DataFrame, n_clusters: int):
        centroids_arr = centroids.to_numpy()
        numerator = 0
        for i in range(n_clusters):
            df_subset = df[df['label'] == i]
            df_subset = df_subset.drop(columns=df_subset.columns[-1])
            numerator += self._cluster_var(df_subset, centroids_arr[i])
        wvc = numerator / df.shape[0]
        return wvc

    def _within_cluster_var_metric(self, df_Q: pd.DataFrame, df_K: pd.DataFrame, centroids_Q, centroids_K, n_clusters):
        wcv_Q = self._within_cluster_var(df_Q, centroids_Q, n_clusters)
        wcv_K = self._within_cluster_var(df_K, centroids_K, n_clusters)
        wcv_metric = (wcv_Q - wcv_K) / wcv_Q
        return wcv_metric

    def cluster_metrics(self, n_clusters: int = 20, downsample_rate=0.2):
        Q_coords_ds = self.Q_coords.sample(
            frac=downsample_rate, random_state=self.random_seed)
        K_coords_ds = self.K_coords.sample(
            frac=downsample_rate, random_state=self.random_seed)

        # Change n_clusters according to how many points are
        # in k_coords_cut relative to k_coords, if the cut has been made.
        n_clusters = round(self.K_keep_percent * n_clusters)

        hcluster_centroids = self._hierarchical_cluster(
            Q_coords_ds, n_clusters=n_clusters)
        q_kmeans_centroids, q_df_labels, q_kmeans = self._kmeans_cluster(df=Q_coords_ds,
                                                                         init=hcluster_centroids,
                                                                         n_clusters=n_clusters)

        k_kmeans_centroids, k_df_labels, k_kmeans = self._kmeans_cluster(df=K_coords_ds,
                                                                         init=q_kmeans_centroids,
                                                                         n_clusters=n_clusters)

        # Prepare for the dict to return
        metrics_dict = {}
        metrics_dict['centroid_distance_n_clusters_'+str(n_clusters)] = self._centroid_distance_metric(
            q_kmeans_centroids, k_kmeans_centroids)
        metrics_dict['cluster_proportion_n_clusters_'+str(n_clusters)] = self._cluster_prop_metric(
            q_df_labels, k_df_labels, n_clusters)
        metrics_dict['iterations_k_n_clusters_'+str(n_clusters)] = k_kmeans.n_iter_
        metrics_dict['wcv_ratio_n_clusters_'+str(n_clusters)] = self._within_cluster_var_metric(
            q_df_labels, k_df_labels, q_kmeans_centroids, k_kmeans_centroids, n_clusters)

        return metrics_dict

    def cut_k(self, partial_type, side):
        '''
        Suppose Q is a partial print
        '''

        assert side in ['R', 'L']
        assert partial_type in ['toe', 'heel', 'inside', 'outside', 'full']

        max_x = max(self.Q_coords.x)
        min_x = min(self.Q_coords.x)
        max_y = max(self.Q_coords.y)
        min_y = min(self.Q_coords.y)

        K_keep = None

        if partial_type == "toe":
            K_keep = self.K_coords[self.K_coords.x < max_x]
        elif partial_type == "heel":
            K_keep = self.K_coords[self.K_coords.x > min_x]
        elif partial_type == "inside":
            if side == "R":
                K_keep = self.K_coords[self.K_coords.y < max_y]
            else:
                K_keep = self.K_coords[self.K_coords.y > min_y]
        elif partial_type == "outside":
            if side == "R":
                K_keep = self.K_coords[self.K_coords.y > min_y]
            else:
                K_keep = self.K_coords[self.K_coords.y < max_y]
        else: # when partial_type == "full"
            K_keep = self.K_coords

        self.K_coords = K_keep
        self.K_keep_percent = len(K_keep) / len(self.K_coords)

    def _dataframe_to_image(self, df, max_x, max_y):
        '''
        Dataframe to image converts a dataframe of aligned coordinates and 
        converts it into an image in the format of a np.ndarray. This is helper 
        function for PC metrics.

        Inputs:
            df: pd.DataFrame
            max_x: int (how wide the image is)
            max_y: int (how tall the image is)

        Returns:
            image (np.ndarray): aligned image in the format of a np.ndarray
        '''
        max_x_int = int(math.ceil(max_x))
        max_y_int = int(math.ceil(max_y))
        image = np.zeros((max_y_int + 1, max_x_int + 1))  # Dimensions based on max x and y coordinates
        image[df['y'].round().astype(int), df['x'].round().astype(int)] = 255  # Set the points to 255 (white) for visualization
        return image

    def _phase_only_correlation(self, image1, image2):
        '''
        Computes phase only correlation between two images using fast fourier 
        transform. 

        Inputs:
            image1: np.ndarray
            image2: np.ndarray

        Returns:
            phase_correlation (float)
        '''
        fft_image1 = np.fft.fft2(image1)
        fft_image2 = np.fft.fft2(image2)

        # Compute cross-power spectrum
        cross_power_spectrum = np.conj(fft_image1) * fft_image2

        # Compute phase correlation
        phase_correlation = np.fft.ifft2(cross_power_spectrum)
        phase_correlation = np.abs(phase_correlation)

        return phase_correlation

    def _calculate_metrics(self, image1, image2):
        '''
        Calculates similarity metrics of two shoeprint images using phase-only 
        correlation. Phase-correlation metrics include phase correlation value, 
        mean squared error, structural similarity index, normalized 
        cross-correlation coefficient, peak-to-sidelobe ratio, and the 
        correlation coefficient.

        Inputs:
            image1: np.ndarray
            image2: np.ndarray

        Returns:
            tuple(float)
        '''
        phase_correlation = self._phase_only_correlation(image1, image2)

        # Find the peak position
        peak_position = np.unravel_index(np.argmax(phase_correlation), 
                                         phase_correlation.shape)

        # Align the images based on the peak position
        aligned_image2 = np.roll(image2, -np.array(peak_position), axis=(0, 1))

        # Mean Squared Error (MSE)
        mse = np.mean((image1 - aligned_image2) ** 2)

        # Structural Similarity Index (SSIM)
        ssim_index, _ = ssim(image1, aligned_image2, full=True)

        # Normalize the images for NCC computation
        norm_image1 = (image1 - np.mean(image1)) / np.std(image1)
        norm_image2 = ((aligned_image2 - np.mean(aligned_image2)) / 
                        np.std(aligned_image2))

        # Normalized Cross-Correlation Coefficient (NCC)
        ncc = np.mean(norm_image1 * norm_image2)

        # Peak-to-Sidelobe Ratio (PSR)
        psr = np.max(phase_correlation) / np.mean(phase_correlation)

        # Correlation Coefficient
        correlation_coefficient = np.corrcoef(image1.ravel(), 
                                              aligned_image2.ravel())[0, 1]
        
        # Stuff we don't know how to explain yet
        peak_value = np.max(phase_correlation)
        
        # peak_position = np.unravel_index(np.argmax(phase_correlation), 
        # phase_correlation.shape)
        return peak_value, mse, ssim_index, ncc, psr, correlation_coefficient

    def pc_metrics(self):
        '''
        Performs phase-correlation calculations and retrieves phase correlation
        metrics.

        Returns:
            (dict): a dictionary object containing all the pc metrics
        '''
        max_x = max(self.Q_coords_full['x'].max(), self.K_coords_full['x'].max())
        max_y = max(self.Q_coords_full['y'].max(), self.K_coords_full['y'].max())
        
        image1 = self._dataframe_to_image(self.Q_coords_full, max_x, max_y)
        image2 = self._dataframe_to_image(self.K_coords_full, max_x, max_y)

        (peak_value, mse_value, ssim_value, ncc_value, psr_value, 
            corr_coeff_value) = self._calculate_metrics(image1, image2)

        metrics_dict = {}
        metrics_dict["peak_value"] = peak_value
        metrics_dict["MSE"] = mse_value
        metrics_dict["SSIM"] = ssim_value
        metrics_dict["NCC"] = ncc_value
        metrics_dict["PSR"] = psr_value
        metrics_dict["correlation_coefficient"] = corr_coeff_value

        return metrics_dict
    
    def jaccard_index(self, round_coords=[0, -1, -2]):
        '''
        Calculates the Jaccard index between the two shoeprints. The Jaccard
        index is the size of the intersection divided by the size of the union
        of the two shoeprints.

        Inputs:
            round_coords (list): list of rounding values to round the coordinates

        Returns:
            (float): Jaccard index
        '''
        metrics_dict = {}
        for r in round_coords:
            set1 = set(map(tuple, self.Q_coords.round(r).values))
            set2 = set(map(tuple, self.K_coords.round(r).values))
            intersection = len(set1.intersection(set2))
            union = len(set1) + len(set2) - intersection
            metrics_dict['jaccard_index_' + str(r)] = intersection / union
        return metrics_dict