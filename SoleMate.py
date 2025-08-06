import streamlit as st
import pandas as pd
import numpy as np
from sole import Sole
from solepair import SolePair
from solepaircompare import SolePairCompare
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from utils.util import WILLIAMS_GOLD, WILLIAMS_PURPLE
import pickle
import zipfile


@st.cache_data
def load_baseline_train():
    return pd.read_csv("static/BASELINE_TRAIN.csv")

@st.cache_data
def load_everything_train():
    return pd.read_csv("static/COMBINED_TRAIN.csv")

@st.cache_resource
def load_baseline_model():
    with open('static/BASELINE_TO_EVERYTHING.pkl', 'rb') as p:
        return pickle.load(p)
    
@st.cache_resource
def load_everything_model():
    with zipfile.ZipFile("static/EVERYTHING_TO_EVERYTHING_NOIND.pkl.zip", 'r') as zip_ref:
        zip_ref.extract("EVERYTHING_TO_EVERYTHING_NOIND.pkl", "static/")
    with open('static/EVERYTHING_TO_EVERYTHING_NOIND.pkl', 'rb') as p:
        return pickle.load(p)


def plot_pre(Q_down, K_down):
    trace_k = go.Scatter(x=K_down.x, y=K_down.y, mode='markers', marker=dict(
        color=WILLIAMS_PURPLE, size=7), name='K')
    trace_q = go.Scatter(x=Q_down.x, y=Q_down.y, mode='markers', marker=dict(
        color=WILLIAMS_GOLD, size=7), name='Q')
    data = [trace_k, trace_q]
    layout = go.Layout(showlegend=True, width=1000,
                       height=600, legend=dict(font=dict(size=32)))
    pre = go.Figure(data=data, layout=layout)
    return pre


def plot_post(Q_down, K_al_down):
    trace_k = go.Scatter(x=K_al_down.x, y=K_al_down.y, mode='markers', marker=dict(
        color=WILLIAMS_PURPLE, size=7), name='K')
    trace_q = go.Scatter(x=Q_down.x, y=Q_down.y, mode='markers', marker=dict(
        color=WILLIAMS_GOLD, size=7), name='Q')
    data = [trace_k, trace_q]
    layout = go.Layout(showlegend=True, width=1000,
                       height=600, legend=dict(font=dict(size=32)))
    post = go.Figure(data=data, layout=layout)
    return post

# Streamlit app


def main():
    Q_file = None
    K_file = None
    # Sidebar
    with st.sidebar:
        model_type = st.selectbox('Select Model', ('Pristine AN (Baseline) Model', 'Full Model'))
        st.divider()
        pair = st.selectbox('Use a preset example pair:',
        # Use example pair
                            ('None', 'Mated Pair #1', 'Mated Pair #2',
                             'Non-Mated Pair #1', 'Non-Mated Pair #2'))
        if pair == 'Mated Pair #1':
            Q_file = "static/mated_2_q.tiff"
            K_file = "static/mated_2_k.tiff"
            q_border_width = 160
            k_border_width = 160
        elif pair == 'Mated Pair #2':
            Q_file = "static/mated_1_q.tiff"
            K_file = "static/mated_1_k.tiff"
            q_border_width = 160
            k_border_width = 160
        elif pair == 'Non-Mated Pair #1':
            Q_file = "static/nonmated_2_q.tiff"
            K_file = "static/nonmated_2_k.tiff"
            q_border_width = 160
            k_border_width = 160
        elif pair == 'Non-Mated Pair #2':
            Q_file = "static/nonmated_1_q.tiff"
            K_file = "static/nonmated_1_k.tiff"
            q_border_width = 160
            k_border_width = 160
        else:
            Q_file = None
            K_file = None

        if not Q_file and not K_file:
            # Upload images and select border widths
            _, c2, _ = st.columns([3,1,3])
            with c2:
                st.header("**OR**")   
            Q_file = st.file_uploader("Upload shoeprint Q", type=[
                                    "png", "jpg", "tiff"])
            q_border_width = st.number_input("Select Q border width (pixels):", 0, 300, 160)
            K_file = st.file_uploader("Upload shoeprint K", type=[
                                    "png", "jpg", "tiff"])
            k_border_width = st.number_input("Select K border width (pixels):", 0, 300, 160)


    col1, col2 = st.columns([1, 1.5])
    with col1:
        _, c2, _ = st.columns([1, 3, 1])
        with c2:
            st.image("static/logo.png")
    with col2:
        st.title("SoleMate")
        st.subheader("An End-To-End System for Forensic Shoeprint Matching")
        st.markdown("Official implementation of \"Improving and Evaluating Machine Learning Methods for Forensic Shoeprint Matching\". ")


    st.divider()

    # Summary
    st.header("Welcome!")
    st.markdown("*SoleMate* is a tool for determining whether or not footwear outsole impressions match. Our algorithm offers a fast\
                and robust method to match the known shoeprint of a suspect (K)\
                to a questioned shoeprint found at the crime scene (Q). We use Iterative\
                Closest Point (ICP), a point cloud registration algorithm, to find the best affine transformation\
                to align two shoeprints. We then calculate metrics and compare them to training data.\
                Using these metrics in a random forest model, we then identify the selected\
                shoeprint pair as mated or non-mated.")
    st.markdown("To use this tool, upload a Q shoeprint and a K shoeprint on\
                the left, or select a preset example shoeprint pair. If the\
                uploaded images come with a frame or ruler, designate the width\
                of the border in pixels. Click the \"Run SoleMate\" button to\
                run the algorithm and see the results! Ensure the prints are dark colored on a light background and adjust the border width (how much of the border to crop out) as necessary. For the Everspry Outsole Scanner, 160 pixels is the proper border width.")

    with st.expander(":athletic_shoe: Introduction to shoeprint pattern matching"):
        st.subheader("Shoeprint Pattern Matching")
        st.markdown("Footwear outsole impressions (shoeprints) are often\
                            found at the scene of a crime and consist of the\
                            marks on a surface created when the materials picked\
                            up by a shoe (e.g., dirt, paint, blood) make contact\
                            with the surface. Shoeprint evidence can be powerful\
                            in connecting an individual to their presence at the\
                            scene of a crime should a known shoe be judged to\
                            match a crime scene print.")
        st.markdown("Characteristics of shoeprints can be broken down\
                            into three categories. Class characteristics include\
                            the size, brand, make, and model of a shoe. Subclass\
                            characteristics consist of smaller differences in\
                            the pattern on the outsole. Individual\
                            characteristics are the traits unique to a single\
                            outsole caused by wear and tear. These\
                            characteristics are known as randomly acquired\
                            characteristics (RACs).")
        st.markdown("Proving that two shoeprints were made by the same\
                            shoe requires that their RACs match. It can be\
                            extremely difficult to detect such small\
                            differences in outsole impressions, even for a\
                            highly trained forensic examiner. We created this\
                            tool to automate the pattern matching process,\
                            reducing the possibility for human error and\
                            allowing us to quantify the degree of similarity\
                            between two shoeprints.")
    if Q_file and K_file:
        if st.sidebar.button("Run SoleMate", type='primary'):
            st.divider()
            # Check if both images are uploaded
            if Q_file and K_file:
                Q = Sole(Q_file, border_width=q_border_width)
                K = Sole(K_file, border_width=k_border_width)
                pair = SolePair(Q, K, True)

                # ICP
                st.header("ICP Alignment")
                st.markdown(
                    "We calculate the best rigid body transformation to align the K shoeprint to the Q shoeprint.")
                with st.spinner("Aligning soles..."):
                    sc = SolePairCompare(pair, icp_downsample_rates=[0.05], two_way=True, shift_left=True,
                                         shift_right=True, shift_down=True, shift_up=True)
                K_down = K.coords.sample(frac=0.1)
                K_al_down = K.aligned_coordinates.sample(frac=0.1)
                Q_down = Q.coords.sample(frac=0.1)

                tab1, tab2 = st.tabs(["Pre-ICP", "Post-ICP"])
                with tab1:
                    pre = plot_pre(Q_down, K_down)
                    st.plotly_chart(pre)
                with tab2:
                    post = plot_post(Q_down, K_al_down)
                    st.plotly_chart(post)

                with st.expander(":question: What is ICP?"):
                    st.subheader("Iterative Closest Point")
                    st.markdown("Aligning two shoeprints might involve rotating and\
                                translating one set of points to match the other,\
                                but it should not involve any sort of stretching,\
                                since we want to maintain the two shoes’ relative\
                                scales. We use a method called iterative closest\
                                point (ICP) to empirically find the angle of\
                                rotation and horizontal and vertical translation\
                                that best align the two point clouds.")
                    st.markdown("To use ICP, we first need to designate one point\
                                cloud as a reference for the other to be shifted.\
                                Then, from the initial alignment of the point\
                                clouds, the algorithm finds the distance between\
                                each point in the shifting cloud to the closest\
                                point in the reference cloud. Using these\
                                distances, we compute a rotation and translation to\
                                minimize the root mean square of the distances.\
                                The algorithm iteratively computes distances and\
                                applies transformations until it determines that a\
                                local minimum root mean square has been reached, at\
                                which point the point clouds should be aligned.")

                st.divider()

                with st.spinner("Calculating metrics..."):
                    # Metrics
                    q_pct_threshold_3 = sc.propn_overlap()
                    k_pct_threshold_3 = sc.propn_overlap(Q_as_base=False)
                    dist_metrics = sc.min_dist()
                    all_cluster_metrics = sc.cluster_metrics(n_clusters=20)
                    all_cluster_metrics.update(sc.cluster_metrics(n_clusters=100))
                    phase_correlation_metrics =sc.pc_metrics()

                    st.header("Metrics")
                    st.markdown("We quantify similarity between shoeprints\
                                using a number of metrics. The dotted line\
                                represents where in the distribution the metric\
                                computed from the input shoeprint pair lies.")

                    # Overlap
                    st.subheader("Overlap")
                    if model_type == 'Pristine AN (Baseline) Model':
                        dataset = load_baseline_train()
                        rf_model = load_baseline_model()
                    else:
                        dataset = load_everything_train()
                        rf_model = load_everything_model()

                    col1, col2 = st.columns(2)
                    with col1:
                        q_pct = plt.figure(figsize=(10, 7))
                        sns.kdeplot(data=dataset, x="q_pct_threshold_3", hue="mated",
                                    fill=True, palette=[WILLIAMS_PURPLE, WILLIAMS_GOLD], alpha=0.6
                                    )
                        plt.axvline(x=q_pct_threshold_3, color='#C86914',
                                    linestyle='--', linewidth=3)
                        plt.text(q_pct_threshold_3, -0.25, 'Test Pair', verticalalignment='bottom',
                                 horizontalalignment='center', fontsize=14, weight='bold', color="#C86914")
                        plt.title("Q Percent Overlap")
                        st.pyplot(q_pct)
                        plt.close(q_pct)
                    with col2:
                        k_pct = plt.figure(figsize=(10, 7))
                        sns.kdeplot(data=dataset, x="k_pct_threshold_3", hue="mated",
                                    fill=True, palette=[WILLIAMS_PURPLE, WILLIAMS_GOLD], alpha=0.6
                                    )
                        plt.axvline(x=k_pct_threshold_3, color='#C86914',
                                    linestyle='--', linewidth=3)
                        plt.text(k_pct_threshold_3, -0.25, 'Test Pair', verticalalignment='bottom',
                                 horizontalalignment='center', fontsize=14, weight='bold', color="#C86914")
                        plt.title("K Percent Overlap")
                        st.pyplot(k_pct)
                        plt.close(k_pct)

                    with st.expander(":bar_chart: About the metric: Overlap"):
                        st.subheader("Percent Overlap")
                        st.markdown("Percent overlap is the proportion of points in one\
                                    shoeprint that are within three pixels of the other\
                                    shoeprint after alignment. We observe the overlap\
                                    in both directions— that is, K on Q and Q on K. A\
                                    high percent overlap indicates a higher likelihood\
                                    of the shoeprints originating from a mated pair. We\
                                    set the threshold at 3 pixels for this example.")

                    # Distance
                    st.subheader("Closest Point Distances")

                    col1, col2 = st.columns(2)
                    with col1:
                        mean = plt.figure(figsize=(10, 7))
                        sns.kdeplot(data=dataset, x="mean", hue="mated",
                                    fill=True, palette=[WILLIAMS_PURPLE, WILLIAMS_GOLD], alpha=0.6
                                    )
                        plt.axvline(x=dist_metrics['mean'], color='#C86914',
                                    linestyle='--', linewidth=3)
                        plt.text(dist_metrics['mean'], -0.015, 'Test Pair', verticalalignment='bottom',
                                 horizontalalignment='center', fontsize=14, weight='bold', color="#C86914")
                        plt.title("Mean CP Distance")
                        plt.xlim((-5, 50))
                        st.pyplot(mean)
                        plt.close(mean)
                    with col2:
                        std = plt.figure(figsize=(10, 7))
                        sns.kdeplot(data=dataset, x="std", hue="mated",
                                    fill=True, palette=[WILLIAMS_PURPLE, WILLIAMS_GOLD], alpha=0.6
                                    )
                        plt.axvline(x=dist_metrics['std'], color='#C86914',
                                    linestyle='--', linewidth=3)
                        plt.text(dist_metrics['std'], -0.0075, 'Test Pair', verticalalignment='bottom',
                                 horizontalalignment='center', fontsize=14, weight='bold', color="#C86914")
                        plt.title("Standard Deviation CP Distance")
                        plt.xlim((-5, 50))
                        st.pyplot(std)
                        plt.close(std)

                    col1, col2 = st.columns(2)
                    with col1:
                        p10 = plt.figure(figsize=(10, 7))
                        sns.kdeplot(data=dataset, x="0.1", hue="mated",
                                    fill=True, palette=[WILLIAMS_PURPLE, WILLIAMS_GOLD], alpha=0.6
                                    )
                        plt.axvline(x=dist_metrics['0.1'], color='#C86914',
                                    linestyle='--', linewidth=3)
                        plt.text(dist_metrics['0.1'], -0.15, 'Test Pair', verticalalignment='bottom',
                                 horizontalalignment='center', fontsize=14, weight='bold', color="#C86914")
                        plt.title("10th Percentile CP Distance")
                        plt.xlim((-0.25, 3))
                        st.pyplot(p10)
                        plt.close(p10)
                    with col2:
                        p25 = plt.figure(figsize=(10, 7))
                        sns.kdeplot(data=dataset, x="0.25", hue="mated",
                                    fill=True, palette=[WILLIAMS_PURPLE, WILLIAMS_GOLD], alpha=0.6
                                    )
                        plt.axvline(x=dist_metrics['0.25'], color='#C86914',
                                    linestyle='--', linewidth=3)
                        plt.text(dist_metrics['0.25'], -0.075, 'Test Pair', verticalalignment='bottom',
                                 horizontalalignment='center', fontsize=14, weight='bold', color="#C86914")
                        plt.title("25th Percentile CP Distance")
                        plt.xlim((-0.5, 10))
                        st.pyplot(p25)
                        plt.close(p25)

                    col1, col2 = st.columns(2)
                    with col1:
                        p50 = plt.figure(figsize=(10, 7))
                        sns.kdeplot(data=dataset, x="0.5", hue="mated",
                                    fill=True, palette=[WILLIAMS_PURPLE, WILLIAMS_GOLD], alpha=0.6
                                    )
                        plt.axvline(x=dist_metrics['0.5'], color='#C86914',
                                    linestyle='--', linewidth=3)
                        plt.text(dist_metrics['0.5'], -0.025, 'Test Pair', verticalalignment='bottom',
                                 horizontalalignment='center', fontsize=14, weight='bold', color="#C86914")
                        plt.title("Median CP Distance")
                        plt.xlim((-2, 30))
                        st.pyplot(p50)
                        plt.close(p50)
                    with col2:
                        p75 = plt.figure(figsize=(10, 7))
                        sns.kdeplot(data=dataset, x="0.75", hue="mated",
                                    fill=True, palette=[WILLIAMS_PURPLE, WILLIAMS_GOLD], alpha=0.6
                                    )
                        plt.axvline(x=dist_metrics['0.75'], color='#C86914',
                                    linestyle='--', linewidth=3)
                        plt.text(dist_metrics['0.75'], -0.01, 'Test Pair', verticalalignment='bottom',
                                 horizontalalignment='center', fontsize=14, weight='bold', color="#C86914")
                        plt.title("75th Percentile CP Distance")
                        plt.xlim((-5, 70))
                        st.pyplot(p75)
                        plt.close(p75)

                    __, col2, __ = st.columns([1, 2, 1])
                    with col2:
                        p90 = plt.figure(figsize=(10, 7))
                        sns.kdeplot(data=dataset, x="0.9", hue="mated",
                                    fill=True, palette=[WILLIAMS_PURPLE, WILLIAMS_GOLD], alpha=0.6
                                    )
                        plt.axvline(x=dist_metrics['0.9'], color='#C86914',
                                    linestyle='--', linewidth=3)
                        plt.text(dist_metrics['0.9'], -0.005, 'Test Pair', verticalalignment='bottom',
                                 horizontalalignment='center', fontsize=14, weight='bold', color="#C86914")
                        plt.title("90th Percentile CP Distance")
                        plt.xlim((-10, 150))
                        st.pyplot(p90)
                        plt.close(p90)

                    with st.expander(":bar_chart: About the metric: Closest Point Distances"):
                        st.subheader("Closest Point Distances")
                        st.markdown("To compute closest point distance metrics, we\
                                    first measure and record the distance between each\
                                    point in the Q shoeprint to the closest point in\
                                    the aligned K shoeprint. Once we have distances\
                                    corresponding to each point in Q, we summarize\
                                    their distribution with the following metrics:\
                                    mean, median, standard deviation, 10th percentile,\
                                    25th percentile, 75th percentile, and 90th\
                                    percentile. For each of these metrics, the smaller\
                                    the magnitude, the more likely the shoeprint pair\
                                    is mated.")

                    # Clustering
                    st.subheader("Clustering")
                    col1, col2 = st.columns(2)
                    with col1:
                        centroid_distance = plt.figure(figsize=(10, 7))
                        sns.kdeplot(data=dataset, x="centroid_distance_n_clusters_20", hue="mated",
                                    fill=True, palette=[WILLIAMS_PURPLE, WILLIAMS_GOLD], alpha=0.6
                                    )
                        plt.axvline(x=all_cluster_metrics['centroid_distance_n_clusters_20'], color='#C86914',
                                    linestyle='--', linewidth=3)
                        plt.text(all_cluster_metrics['centroid_distance_n_clusters_20'], -0.001, 'Test Pair', verticalalignment='bottom',
                                 horizontalalignment='center', fontsize=14, weight='bold', color="#C86914")
                        plt.title("Centroid Distance")
                        plt.xlim((-50, 600))
                        st.pyplot(centroid_distance)
                        plt.close(centroid_distance)
                    with col2:
                        cluster_proprtion = plt.figure(figsize=(10, 7))
                        sns.kdeplot(data=dataset, x="cluster_proportion_n_clusters_20", hue="mated",
                                    fill=True, palette=[WILLIAMS_PURPLE, WILLIAMS_GOLD], alpha=0.6
                                    )
                        plt.axvline(x=all_cluster_metrics['cluster_proportion_n_clusters_20'], color='#C86914',
                                    linestyle='--', linewidth=3)
                        plt.text(all_cluster_metrics['cluster_proportion_n_clusters_20'], -5, 'Test Pair', verticalalignment='bottom',
                                 horizontalalignment='center', fontsize=14, weight='bold', color="#C86914")
                        plt.title("Cluster Proportion")
                        st.pyplot(cluster_proprtion)
                        plt.close(cluster_proprtion)

                    col1, col2 = st.columns(2)
                    with col1:
                        iterations_k_n_clusters_20 = plt.figure(figsize=(10, 7))
                        sns.kdeplot(data=dataset, x="iterations_k_n_clusters_20", hue="mated",
                                    fill=True, palette=[WILLIAMS_PURPLE, WILLIAMS_GOLD], alpha=0.6
                                    )
                        plt.axvline(x=all_cluster_metrics['iterations_k_n_clusters_20'], color='#C86914',
                                    linestyle='--', linewidth=3)
                        plt.text(all_cluster_metrics['iterations_k_n_clusters_20'], -0.004, 'Test Pair', verticalalignment='bottom',
                                 horizontalalignment='center', fontsize=14, weight='bold', color="#C86914")
                        plt.title("Iterations K")
                        st.pyplot(iterations_k_n_clusters_20)
                        plt.close(iterations_k_n_clusters_20)
                    with col2:
                        wcv_ratio_n_clusters_20 = plt.figure(figsize=(10, 7))
                        sns.kdeplot(data=dataset, x="wcv_ratio_n_clusters_20", hue="mated",
                                    fill=True, palette=[WILLIAMS_PURPLE, WILLIAMS_GOLD], alpha=0.6
                                    )
                        plt.axvline(x=all_cluster_metrics['wcv_ratio_n_clusters_20'], color='#C86914',
                                    linestyle='--', linewidth=3)
                        plt.text(all_cluster_metrics['wcv_ratio_n_clusters_20'], -0.25, 'Test Pair', verticalalignment='bottom',
                                 horizontalalignment='center', fontsize=14, weight='bold', color="#C86914")
                        plt.title("Within Cluster Variation")
                        plt.xlim((-2, 0.5))
                        st.pyplot(wcv_ratio_n_clusters_20)
                        plt.close(wcv_ratio_n_clusters_20)

                    with st.expander(":bar_chart: About the metric: Clustering"):
                        st.subheader("Clustering")
                        st.markdown("Calculating our clustering metrics relies on two\
                                    different clustering algorithms. Hierarchical\
                                    clustering requires a predetermined number of\
                                    clusters, k, and a linkage function. We opt to use\
                                    the Ward linkage function which minimizes the\
                                    variance of the Euclidean distances between points\
                                    within each of the k clusters. k-means clustering,\
                                    on the other hand, is initialized with the\
                                    coordinates of cluster centroids. Using these\
                                    centroid coordinates as a prototype, the k-means\
                                    algorithm minimizes the within-cluster\
                                    sum-of-squares (known as inertia) to find as many\
                                    clusters as centroid coordinates were inputted.")
                        st.markdown("Our implementation of clustering begins with\
                                    hierarchical clustering on the Q shoeprint, and we\
                                    return the centroids of the clusters created. We\
                                    then use these centroids to initialize k-means\
                                    clusters once again on the Q shoeprint. The\
                                    clusters change slightly, so we return the updated\
                                    cluster centroids. We then run k-means clustering\
                                    on the K shoeprint with these centroids. We use the\
                                    similarities between the k-means clusters of Q and\
                                    K to quantify the similarity between the two\
                                    shoeprints. Our measures of similarity are the root\
                                    mean squared of the differences between cluster\
                                    sizes as a proportion of the number of points in\
                                    the entire print, the root mean squared of the\
                                    distances between the centroids of clusters in Q\
                                    and the corresponding updated centroids of the\
                                    clusters in K, the difference between the within-cluster\
                                    variation in Q and K scaled by the within-cluster\
                                    variation in Q, and the number of\
                                    iterations k-means clustering took to find clusters\
                                    in K. For each of the clustering metrics,\
                                    smaller magnitudes are indicative of mated pairs.\
                                    We set the number of clusters to 20 for this example.")
                    
                    # Phase Correlation
                    st.subheader("Phase Correlation")
                    col1, col2 = st.columns(2)
                    # with col1:
                    #     peak_value = plt.figure(figsize=(10, 7))
                    #     sns.kdeplot(data=dataset, x="peak_value", hue="mated",
                    #                 fill=True, palette=[WILLIAMS_PURPLE, WILLIAMS_GOLD], alpha=0.6
                    #                 )
                    #     # TODO: Fix this
                    #     plt.axvline(x=phase_correlation_metrics['peak_value'], color='#C86914',
                    #                 linestyle='--', linewidth=3)
                    #     # plt.text(phase_correlation_metrics['peak_value'], 0, 'Test Pair', verticalalignment='bottom',
                    #     #          horizontalalignment='center', fontsize=14, weight='bold', color="#C86914")
                    #     plt.title("Peak Value")
                    #     plt.xlim((-50, 600))
                    #     st.pyplot(peak_value)
                    #     plt.close(peak_value)
                    with col1:
                        PSR = plt.figure(figsize=(10, 7))
                        sns.kdeplot(data=dataset, x="PSR", hue="mated",
                                    fill=True, palette=[WILLIAMS_PURPLE, WILLIAMS_GOLD], alpha=0.6
                                    )
                        plt.axvline(x=phase_correlation_metrics['PSR'], color='#C86914',
                                    linestyle='--', linewidth=3)
                        plt.text(phase_correlation_metrics['PSR'], -0.01, 'Test Pair', verticalalignment='bottom',
                                 horizontalalignment='center', fontsize=14, weight='bold', color="#C86914")
                        plt.title("Peak-to-Sidelobe Ratio (PSR)")
                        st.pyplot(PSR)
                        plt.close(PSR)    

                    with st.expander(":bar_chart: About the metric: Phase Correlation"):
                        st.subheader("Phase Correlation")
                        st.markdown("The Fourier transform is a mathematical tool\
                                    that can be utilized for signal processing\
                                    and image analysis. We perform phase correlation\
                                    analysis using the full shoeprint image prior to edge detection.\
                                    The transformation matrix calculated \
                                    during ICP alignment is saved and used to align\
                                    the full coordinates. In our implementation\
                                    of phase correlation, we follow the algorithm\
                                     first implemented by Kuglin and Hines")
                        st.markdown("Peak value, corresponds to the maximum\
                                     value of the phase correlation matrix. PSR\
                                     reflects the relative strength of the phase\
                                     correlation peak with its sidelobe levels")
                

                st.divider()

                st.header("Classification")
                st.markdown(
                    "Based on a trained random forest model, we hypothesize whether \
                        the input shoeprint pair is mated or non-mated.")
                with st.spinner("Classifying pair..."):
                    # Creating a Dataframe row containing the new metrics
                    row = {'q_pct_threshold_1': sc.propn_overlap(threshold=1),
                        'k_pct_threshold_1': sc.propn_overlap(threshold=1, Q_as_base=False),
                        'q_pct_threshold_2': sc.propn_overlap(threshold=2),
                        'k_pct_threshold_2': sc.propn_overlap(threshold=2, Q_as_base=False),
                        'q_pct_threshold_3': q_pct_threshold_3,
                        'k_pct_threshold_3': k_pct_threshold_3,
                        'q_pct_threshold_5': sc.propn_overlap(threshold=5),
                        'k_pct_threshold_5': sc.propn_overlap(threshold=5, Q_as_base=False),
                        'q_pct_threshold_10': sc.propn_overlap(threshold=10),
                        'k_pct_threshold_10': sc.propn_overlap(threshold=10, Q_as_base=False),
                        'q_points_count': Q.coords.shape[0],
                        'k_points_count': K.coords.shape[0]
                        }
                    row.update(dist_metrics)
                    row.update(all_cluster_metrics)
                    row.update(phase_correlation_metrics)
                    row.update(sc.jaccard_index())
                    row = pd.DataFrame(row, index=[0])

                # Subsetting relevant features
                row = row[list(rf_model.feature_names_in_)].round(2)
                # Indexing into rf probability for class=1 i.e. mated=True
                score = rf_model.predict_proba(row)[0][1]

                mated = "Mated" if score > 0.5 else "Non-Mated"
                prob = round(score, 2) if mated=="Mated" else round(1-score, 2)
                st.success(
                    f"Our model predicts that the shoeprints are **_{mated}_**", icon="👟")
                st.markdown("A summary of all the metrics we calculated:")
                st.dataframe(row)
                st.markdown(f"RF posterior probability: **{prob}**")

                with st.expander(":question: What is random forest?"):
                    st.subheader("Random Forest")
                    st.markdown("Random forest is a machine learning algorithm that\
                                takes in several variables and returns a number\
                                between 0 and 1 representing a probability outcome.\
                                A random forest is trained with a set of data and \
                                employs many decision trees (hence, a forest) to \
                                learn how to best predict the binary outcome (in \
                                this case, mated or non-mated) based on the \
                                variables available and determines which variables \
                                are more important than others in predicting the \
                                outcome. We trained our model on the similarity \
                                metrics from thousands of known mated and non-mated \
                                shoeprint pairs, \
                                and the model will assess based on this training the \
                                probability that the input pair is mated or \
                                non-mated.")

                with st.expander(":technologist: Our random forest implementation"):
                    st.subheader("Our Random Forest Implementation")
                    st.markdown("We trained our random forest on data from \
                                [this dataset](https://forensicstats.org/shoeoutsoleimpressionstudy/).\
                                To create known mated pairs, we selected different\
                                scans from the same shoe taken at the same time, and\
                                to create non-mated pairs, we selected scans from\
                                different shoes of the same make, model, and size to\
                                simulate similar shoes with different randomly\
                                acquired characteristics. We trained our random\
                                forest on 70\% of these data and tested it with the\
                                remaining completely independent 30\% (i.e., no\
                                image appears in both the training and test set).\
                                See the variable importance of the random forest\
                                model below.")
                    # Variable importance plot for random forest
                    importances = rf_model.feature_importances_
                    feature_names = ['distance '+metric if metric in ['0.1', '0.25', '0.5', '0.75',
                                                                        '0.9', 'mean', 'std'] else metric for metric in rf_model.feature_names_in_]
                    feature_importances = pd.DataFrame(
                        {'Feature': feature_names, 'Importance': importances})
                    feature_importances = feature_importances.sort_values(
                        'Importance', ascending=True)
                    fig = px.bar(feature_importances, x='Importance',
                                    y='Feature', orientation='h', color='Importance', color_continuous_scale=["#B1008E", "#500082"],
                                    width=820,
                                    height=600)
                    fig.update_layout(yaxis=dict(tickfont=dict(size=15)))
                    fig.update(layout_coloraxis_showscale=False)
                    st.plotly_chart(fig)

    st.divider()
    st.markdown(
        "*Disclaimer: This tool should be used for research and education purposes only.*")
    st.markdown(
        "Developed and maintained by Simon Angoluan, Divij Jain, Saatvik Kher, Lena Liang, Yufeng Wu, and Ashley Zheng.")
    st.markdown(
        "We conducted our research in collaboration with the [Center for Statistics and Applications in Forensic Evidence](https://forensicstats.org/).")
    st.markdown(
        "Please [reach out to us](mailto:saatvikkher1@gmail.com) if you run into any issues or have any comments or questions.")


# Run the app
if __name__ == "__main__":
    st.set_page_config(page_title='SoleMate', page_icon = 'logo.png', layout='wide')
    css = '''
    <style>
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size:16pt;
        }
    </style>
    '''
    st.markdown(css, unsafe_allow_html=True)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    main()
