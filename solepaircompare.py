import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.spatial import KDTree
import math
from solepair import SolePair
from sklearn.cluster import AgglomerativeClustering, KMeans
from skimage.metrics import structural_similarity as ssim

class SolePairCompare:
    '''
    SolePairCompare will take two shoeprint images (Q and K) and eventually 
    outputs the collection of similarity metrics comparing Q and K including 
    variations of proportion overlap and overlap, metrics based on clustering, 
    metrics based on the closest point in the other shoe, metrics based on 
    phase-only correlation, and image-based metrics. SolePairCompare initializes
    the comparison class by aligning the two shoeprints in the given pair.

    This class also provides methods for intermediate steps in this pipeline.
    '''

    def __init__(self,
                 pair: SolePair,
                 downsample_rate=1.0,
                 icp_downsample_rates=None,
                 random_seed=0,
                 shift_left=False,
                 shift_right=False,
                 shift_up=False,
                 shift_down=False,
                 two_way=False,
                 icp_overlap_threshold=3) -> None:
        '''
        Inputs:
            pair (SolePair): the pair of shoes used for comparison
            downsample_rate (float): a real number between 0 and 1 by which the 
                points in the shoe will be downsampled for alignment
            random_seed (int)
            shift_left (bool): try the left random-start in ICP implementation
            shift_right (bool): try the right random-start in ICP implementation
            shift_up (bool): try the up random-start in ICP implementation
            shift_down (bool): try the down random-start in ICP implementation
            two_way (bool): try aliging the shoes Q over K and K over Q in ICP
            icp_overlap_threshold (float): threshold for calculating proportion 
                overlap when determing the best alignment direction
        '''

        # None-safe default (mutable defaults are shared across calls) and
        # sort a copy so the caller's list is never mutated in place.
        if icp_downsample_rates is None or len(icp_downsample_rates) == 0:
            icp_downsample_rates = [1.0]
        icp_downsample_rates = sorted(icp_downsample_rates)
            
        best_icp_downsample_rate = None
        best_propn_overlap = -1

        for icp_downsample_rate in icp_downsample_rates:
            self._Q_coords, self._K_coords, self._Q_coords_full, self._K_coords_full = pair.icp_transform(downsample_rate=icp_downsample_rate,
                                                                                                         shift_left=shift_left,
                                                                                                         shift_right=shift_right,
                                                                                                         shift_up=shift_up,
                                                                                                         shift_down=shift_down,
                                                                                                         two_way=two_way,
                                                                                                         overlap_threshold=icp_overlap_threshold)

            self._Q_coords = self._Q_coords.sample(frac=downsample_rate, 
                                                   random_state=random_seed)
            self._K_coords = self._K_coords.sample(frac=downsample_rate, 
                                                   random_state=random_seed)
            
            # Check the proportion overlap of this icp_downsample_rate
            po = self.propn_overlap()

            # Choose the higher proportion overlap as the better ICP 
            if po > best_propn_overlap:
                best_propn_overlap = po
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

        # The proportion of original K that will be kept
        self.K_keep_propn = 1.0
        self.pair = pair
        self.random_seed = random_seed
    
    @property
    def Q_coords(self) -> pd.DataFrame:
        '''Getter method for Q dataframe of coordinates'''
        return self._Q_coords

    @property
    def K_coords(self) -> pd.DataFrame:
        '''Getter method for K dataframe of coordinates'''
        return self._K_coords

    @property
    def Q_coords_full(self) -> pd.DataFrame:
        '''Getter method for Q full dataframe of coordinates'''
        return self._Q_coords_full

    @property
    def K_coords_full(self) -> pd.DataFrame:
        '''Getter method for K full dataframe of coordinates'''
        return self._K_coords_full
    
    @Q_coords.setter
    def Q_coords(self, value) -> None:
        '''Setter method for Q dataframe of coordinates'''
        self._Q_coords = value
    
    @K_coords.setter
    def K_coords(self, value) -> None:
        '''Setter method for K dataframe of coordinates'''
        self._K_coords = value

    @Q_coords_full.setter
    def Q_coords_full(self, value) -> None:
        '''Setter method for Q full dataframe of coordinates'''
        self._Q_coords_full = value

    @K_coords_full.setter
    def K_coords_full(self, value) -> None:
        '''Setter method for K full dataframe of coordinates'''
        self._K_coords_full = value

    def _df_to_hash(self, df):
        '''
        Changes a dataframe to a hashtable.

        Inputs:
            df (pd.DataFrame of integer values)

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
            '''
            A helper function for _is_overlap that retrieves the distance b/w
            point_a and point_b

            Inputs:
                point_a (list(float))
                point_b (list(float))

            Returns: (float)
            '''
            x1, y1 = point_a[0], point_a[1]
            x2, y2 = point_b[0], point_b[1]
            return math.hypot(x1-x2, y1-y2)

        for potential_x in range(x-threshold, x+threshold+1):
            for potential_y in range(y-threshold, y+threshold+1):
                if potential_x in ht and potential_y in ht[potential_x]:
                    # check whether the potential point is within threshold
                    if get_dist([x, y], [potential_x, potential_y]) <= threshold:
                        return True
        return False

    def propn_overlap(self, Q_as_base=True, threshold=3):
        '''
        (Similarity metric) 

        Calculates the proportion of points in the base shoeprint (determined by
        Q_as_base) that overlap with points in the other shoeprint.

        Inputs: 
            Q_as_base (bool): if Q is the base dataframe, defaults to True
            threshold (int): if distance between two points <= threshold, 
                             it qualifies as an overlap. 

        Returns: (float): proportion of points that overlap 
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

        # Guard: an empty base frame (e.g. downsample_rate that keeps 0
        # rows) makes apply() return a Series under pandas 3.0, crashing
        # the caller with "truth value of a Series is ambiguous".
        if len(df1) == 0:
            return 0.0

        overlap_count = df1.apply(lambda point: self._is_overlap(
            point['x'], point['y'], ht, threshold), axis=1).sum()
        return float(overlap_count) / len(df1)

    def min_dist(self, Q_as_base=True):
        '''
        (Similarity metric) 

        min_dist is a similarity metric comparing the dataframes of K and Q. 
        For each sample point from the base dataframe, min_dist finds the 
        Euclidean distance of the closest point in the other dataframe using the 
        kdtree closest neighbor algorithm. 

        It outputs distribution statistics including the average, 
        standard deviation, and 0.1, 0.25, 0.5, 0.75, 0.9 quantile values.

        Inputs:
            Q_as_base (bool): if Q is the base dataframe, defaults to True

        Returns:
            (dict): a dictionary of statistics with keys "mean", "std", "0.1", 
                    "0.25", "0.5", "0.75", "0.9", "kurtosis" and values of float
                    type
        '''
        # If Q_as_base == True, Q is the shoe we sample from
        if Q_as_base:
            df1 = self.K_coords
            df2 = self.Q_coords
        else:
            df1 = self.Q_coords
            df2 = self.K_coords

        min_dists = []

        # We assume the column names are "x" and "y" as created in Sole
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

        return min_dists_dict

    def _hierarchical_cluster(self, df: pd.DataFrame, n_clusters: int) -> pd.DataFrame:
        '''
        This function takes in pandas DataFrame and runs hierarchical clustering
        on them. It returns the centroid points of the hierarchical clustering.

        Inputs:
            df (pd.DataFrame): shoeprint used for clustering
            n_clusters (int): the number of clusters
        
        Returns:
            centroid_pts (pd.dataframe with columns 'x' and 'y')
        '''
        df_arr = df.to_numpy()
        df_label = df.copy(deep=True)
        hierarchical_cluster = AgglomerativeClustering(n_clusters=n_clusters,
                                                       metric='euclidean')
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
            df (pd.DataFrame): x, y coordinates of a shoe
            init (pd.DataFrame): the points where kmeans clustering begins
            n_clusters (int): the number of clusters

        Returns:
            centroids (pd.DataFrame)
            df_labels (pd.DataFrame with columns "x", "y", and "label"): a df of
                labels corresponding to the cluster of each (x, y) point
            kmeans (KMeans object): A summary of kmeans clustering for metrics
        '''
        assert init.shape[0] == n_clusters

        df_arr = df.to_numpy()
        df_labels = df.copy(deep=True)
        kmeans = KMeans(n_clusters=n_clusters, init=init, n_init=1)
        kmeans.fit(df_arr)

        df_labels['label'] = kmeans.labels_
        cluster_centers = kmeans.cluster_centers_
        centroids = pd.DataFrame(cluster_centers, columns=['x', 'y'])

        return centroids, df_labels, kmeans

    def _centroid_distance_metric(self, centroids_a: pd.DataFrame, centroids_b: pd.DataFrame) -> float:
        '''
        Computes the centroid distance metric. Given two sets of centroids, 
        compute the distance between each paired centroid. Then, compute the 
        rmse of these distances. Since rmse is always postive, centroids_a can 
        be either shoeprint Q or shoeprint K.

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
        The cluster proportion metric is the rmse of the difference of 
        proportion sizes between corresponding clusters in shoeprint Q and 
        shoeprint K. Since rmse is claculated, df_a can be either shoeprint Q or 
        shoeprint K

        Inputs:
            df_a (pd.DataFrame): dataframe of either shoeprint Q or shoeprint K
            df_b (pd.DataFrame): dataframe of the other shoeprint
            n_clusters (int): the number of clusters

        Returns:
            rmse (float): the rmse of cluster proportion sizes
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

    def _cluster_var(self, df: pd.DataFrame, centroid: np.ndarray) -> float:
        '''
        Calculates the sum of distances squared for a singular cluster.

        Inputs:
            df (pd.DataFrame): The shoeprint point-cloud without labels for clusters.
            centroid (np.ndarray): The centroids to compare distances to

        Returns:
            (float): The sum of squared distances between data points in a 
                cluster and its centroid
        '''
        df_arr = df.to_numpy()
        distance = np.linalg.norm(df_arr - centroid, axis=1)
        distance_squared = distance ** 2
        sum_distance_squared = np.sum(distance_squared)
        return sum_distance_squared

    def _within_cluster_var(self, df: pd.DataFrame, centroids: pd.DataFrame, n_clusters: int):
        '''
        Calculate the average within-cluster variation across all clusters.

        Inputs:
            df (pd.DataFrame): The point-cloud data frame with cluster labels
            centroids (pd.DataFrame): A dataframe with n_clusters rows and x,y
                coordinates corresponding to the cluster number
            n_clusters (int): The number of clusters
        
        Returns: 
            (float): within cluster variation computation
        '''
        centroids_arr = centroids.to_numpy()
        numerator = 0
        for i in range(n_clusters):
            df_subset = df[df['label'] == i]
            df_subset = df_subset.drop(columns=df_subset.columns[-1])
            numerator += self._cluster_var(df_subset, centroids_arr[i])
        wvc = numerator / df.shape[0]
        return wvc

    def _within_cluster_var_metric(self, df_Q: pd.DataFrame, df_K: pd.DataFrame, centroids_Q: pd.DataFrame, 
                                   centroids_K: pd.DataFrame, n_clusters: int):
        '''
        Combines the within-cluster-variation of Q and K into one metric by 
        taking the difference between the within-cluster-variation of Q and K 
        and normalizing by the within-cluster-variation of Q.

        Inputs:
            df_Q (pd.DataFrame): the point-cloud of Q with cluster labels
            df_K (pd.DataFrame): the point-cloud of K with cluster labels
            centroids_Q (pd.DataFrame): dataframe of x,y coordinates of the centroids of clusters in Q
            centroids_K (pd.DataFrame): dataframe of x,y coordinates of the centroids of clusters in K
            n_clusters (int): the number of clusters

        Returns:
            (float): the within-cluster-variation metric
        '''
        wcv_Q = self._within_cluster_var(df_Q, centroids_Q, n_clusters)
        wcv_K = self._within_cluster_var(df_K, centroids_K, n_clusters)
        # Guard: identical coincident points give wcv_Q == 0 -> division by zero.
        if wcv_Q == 0:
            return 0.0 if wcv_K == 0 else float('inf')
        wcv_metric = (wcv_Q - wcv_K) / wcv_Q
        return wcv_metric

    def cluster_metrics(self, n_clusters: int = 20, downsample_rate=0.2, n_points_per_cluster=None):
        '''
        Executes clustering then computes similarity metrics based on clustering.

        We compute clustering deterministically by using hierarchical clustering on shoeprint Q.
        Then, the centroids from hierarchical clustering are then used as starting points for 
        k-means clustering in Q. These centroids are then used for k-means clustering in K. We 
        then compare these results to obtain similarity metrics based on clustering. Clustering 
        size can be determined by using a fixed number of clusters regardless of point-cloud 
        size, or choosing how big each cluster can be.

        Metrics include:
            Centroid Distance: The rmse of distances between corresponding 
                centroids.
            Cluster Proportion: The rmse of cluster proportions.
            Iterations K: The number of iterations it took to converge from the 
                Q centroids to k-means clustering in K.
            Within-Cluster-Variation Ratio: The ratio of within cluster 
                variation between Q and K.

        Inputs:
            n_clusters (int): the number of clusters, defaults to 20
            downsample_rate (float): rate of downsample to speed up clustering, defaults to 0.2
            n_points_per_cluster (int or None): number of points per cluster if points-per-cluster 
                are used, defaults to None (cluster number will be determineed by n_clsuters)
        
        Returns:
            (dict): dictionary containing similarity metrics based on clustering
        '''
        Q_coords_ds = self.Q_coords.sample(
            frac=downsample_rate, random_state=self.random_seed)
        K_coords_ds = self.K_coords.sample(
            frac=downsample_rate, random_state=self.random_seed)

        # Cap the clustering sample: hierarchical clustering is O(n^2) in
        # memory; a dark/busy upload can produce hundreds of thousands of
        # points and exhaust RAM (verified MemoryError at ~20k points).
        CLUSTER_SAMPLE_CAP = 15000  # just above the training regime's max sample
        if len(Q_coords_ds) > CLUSTER_SAMPLE_CAP:
            Q_coords_ds = Q_coords_ds.sample(CLUSTER_SAMPLE_CAP,
                                             random_state=self.random_seed)
        if len(K_coords_ds) > CLUSTER_SAMPLE_CAP:
            K_coords_ds = K_coords_ds.sample(CLUSTER_SAMPLE_CAP,
                                             random_state=self.random_seed)

        # Change n_clusters according to how many points are
        # in k_coords_cut relative to k_coords, if the cut has been made.
        # If no cut has been made, n_clusters will not change.
        n_clusters = round(self.K_keep_propn * n_clusters)

        # Guard: clustering needs at least n_clusters points. Raising (rather
        # than clamping) keeps the metric names stable — the trained models
        # look up exact keys like 'centroid_distance_n_clusters_20'.
        if n_clusters < 2 or len(Q_coords_ds) < n_clusters or len(K_coords_ds) < n_clusters:
            raise ValueError(
                f"Not enough points to compute clustering metrics: "
                f"{len(Q_coords_ds)} Q / {len(K_coords_ds)} K points available "
                f"but {n_clusters} clusters are required. Use a clearer, "
                f"fuller shoeprint image.")

        hcluster_centroids = self._hierarchical_cluster(
            Q_coords_ds, n_clusters=n_clusters)
        q_kmeans_centroids, q_df_labels, _ = self._kmeans_cluster(df=Q_coords_ds,
                                                                         init=hcluster_centroids,
                                                                         n_clusters=n_clusters)

        k_kmeans_centroids, k_df_labels, k_kmeans = self._kmeans_cluster(df=K_coords_ds,
                                                                         init=q_kmeans_centroids,
                                                                         n_clusters=n_clusters)

        # Prepare for the dict to return
        if n_points_per_cluster is None:
            metric_name = 'n_clusters_' + str(n_clusters)
        else:
            metric_name = 'n_points_per_cluster_' + str(n_points_per_cluster)

        metrics_dict = {}
        metrics_dict['centroid_distance_'+metric_name] = self._centroid_distance_metric(
            q_kmeans_centroids, k_kmeans_centroids)
        metrics_dict['cluster_proportion_'+metric_name] = self._cluster_prop_metric(
            q_df_labels, k_df_labels, n_clusters)
        metrics_dict['iterations_k_'+metric_name] = k_kmeans.n_iter_
        metrics_dict['wcv_ratio_'+metric_name] = self._within_cluster_var_metric(
            q_df_labels, k_df_labels, q_kmeans_centroids, k_kmeans_centroids, n_clusters)

        return metrics_dict

    def cut_k(self, partial_type, side):
        '''
        Suppose Q is a simulated partial print. This function will cut K based 
        on the way Q is cut. This function modifies the coordinates of K in 
        place and has no returns.

        Inputs:
            partial_type (str): The type of cut Q is.
            side (str): Whether Q and K are right ('R') or left ('L') shoeprints.
        '''
        assert partial_type in ['toe', 'heel', 'inside', 'outside', 'full']
        assert side in ['R', 'L']
        
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

        # Proportion must be measured BEFORE self.K_coords is replaced,
        # otherwise the denominator is the kept subset and the ratio is
        # always 1.0 (silently disabling the n_clusters scaling).
        self.K_keep_propn = (len(K_keep) / len(self.K_coords)
                             if len(self.K_coords) else 0.0)
        self.K_coords = K_keep

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

        # Dimensions based on max x and y coordinates
        image = np.zeros((max_y_int + 1, max_x_int + 1)) 

        # Set the points to 255 (white) for visualization
        image[df['y'].round().astype(int), df['x'].round().astype(int)] = 255  
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
            image1 (np.ndarray): first shoeprint image
            image2 (np.ndarray): second shoeprint image

        Returns:
            tuple(float): similarity metrics
        '''
        # Compute phase correlation and align images
        phase_correlation = self._phase_only_correlation(image1, image2)
        peak_position = np.unravel_index(np.argmax(phase_correlation), 
                                         phase_correlation.shape)
        aligned_image2 = np.roll(image2, -np.array(peak_position), axis=(0, 1))

        # Mean Squared Error (MSE)
        mse = np.mean((image1 - aligned_image2) ** 2)

        # Structural Similarity Index (SSIM)
        # skimage requires the window to fit inside the image; tiny or
        # odd-aspect uploads can rasterize smaller than the default 7x7.
        win = min(7, image1.shape[0], image1.shape[1])
        if win % 2 == 0:
            win -= 1
        ssim_index, _ = ssim(image1, aligned_image2, full=True,
                             data_range=255.0, win_size=win)

        # Peak-to-Sidelobe Ratio (PSR)
        psr = np.max(phase_correlation) / np.mean(phase_correlation)

        # Normalized Correlation Coefficient
        NCC = np.corrcoef(image1.ravel(), aligned_image2.ravel())[0, 1]

        # Sanitize degenerate cases: constant/all-zero images make these
        # NaN (0/0 or zero variance). sklearn accepts NaN and silently
        # biases the prediction, so substitute a neutral 0 instead.
        psr = float(psr) if np.isfinite(psr) else 0.0
        NCC = float(NCC) if np.isfinite(NCC) else 0.0

        # Peak Value
        peak_value = np.max(phase_correlation)

        return peak_value, mse, ssim_index, psr, NCC

    def pc_metrics(self):
        '''
        Performs phase-correlation calculations and retrieves phase correlation
        metrics as well as similarity metrics computed using the entire shoeprint image.

        Returns:
            (dict): a dictionary object containing all the pc metrics
        '''
        # Shift both clouds into a non-negative frame so rasterization is
        # exact: ICP rotation/translation can push coordinates below zero,
        # and negative indices would silently wrap (corrupting the metrics).
        shift_x = min(0.0, float(min(self.Q_coords_full['x'].min(),
                                     self.K_coords_full['x'].min())))
        shift_y = min(0.0, float(min(self.Q_coords_full['y'].min(),
                                     self.K_coords_full['y'].min())))

        Q_shifted = self.Q_coords_full.copy()
        K_shifted = self.K_coords_full.copy()
        Q_shifted['x'] = Q_shifted['x'] - shift_x
        Q_shifted['y'] = Q_shifted['y'] - shift_y
        K_shifted['x'] = K_shifted['x'] - shift_x
        K_shifted['y'] = K_shifted['y'] - shift_y

        max_x = max(Q_shifted['x'].max(), K_shifted['x'].max())
        max_y = max(Q_shifted['y'].max(), K_shifted['y'].max())

        image1 = self._dataframe_to_image(Q_shifted, max_x, max_y)
        image2 = self._dataframe_to_image(K_shifted, max_x, max_y)

        (peak_value, mse_value, ssim_value, psr_value, NCC) = self._calculate_metrics(image1, image2)

        metrics_dict = {}
        metrics_dict["peak_value"] = peak_value
        metrics_dict["MSE"] = mse_value
        metrics_dict["SSIM"] = ssim_value
        metrics_dict["PSR"] = psr_value
        metrics_dict["NCC"] = NCC

        return metrics_dict
    
    def jaccard_index(self, round_coords=[0, -1, -2]):
        '''
        Calculates the Jaccard index between the two shoeprints. The Jaccard
        index is the size of the intersection divided by the size of the union
        of the two shoeprints.

        Inputs:
            round_coords (list): list of rounding values to round the coordinates

        Returns:
            (dict): a dictionary object containing all the jaccard index metrics
            at the given rounding values
        '''
        metrics_dict = {}
        for r in round_coords:
            set1 = set(map(tuple, self.Q_coords.round(r).values))
            set2 = set(map(tuple, self.K_coords.round(r).values))
            intersection = len(set1.intersection(set2))
            union = len(set1) + len(set2) - intersection
            # Guard: degenerate clouds (no points) would divide by zero.
            metrics_dict['jaccard_index_' + str(r)] = intersection / union if union > 0 else 0.0
        return metrics_dict
