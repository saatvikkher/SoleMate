import streamlit as st
import pandas as pd
import numpy as np
from process_util import process_image
from sole import Sole
from solepair import SolePair
from solepaircompare import SolePairCompare
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from util import WILLIAMS_GOLD, WILLIAMS_PURPLE
import pickle

@st.cache_data
def load_train():
    return pd.read_csv("old_results/0711/result_train_0711.csv")

@st.cache_resource
def load_model():
    with open('full.pkl', 'rb') as p:
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
        # Upload images and select border widths
        Q_file = st.file_uploader("Upload shoeprint Q", type=["png", "jpg", "tiff"])
        q_border_width = st.number_input("Select Q border width:", 0, 300, 0)
        K_file = st.file_uploader("Upload shoeprint K", type=["png", "jpg", "tiff"])
        k_border_width = st.number_input("Select K border width:", 0, 300, 0)

        # Use example pair
        st.divider()
        st.markdown("**OR**")

        pair = st.selectbox('Use a preset example pair:',
                                ('None', 'Mated Pair #1', 'Mated Pair #2', 
                                'Non-Mated Pair #1', 'Non-Mated Pair #2'))
        if pair == 'Mated Pair #1':
            Q_file = "example_shoeprints/mated_1_q.tiff"
            K_file = "example_shoeprints/mated_1_k.tiff"
            q_border_width = 160
            k_border_width = 160
        elif pair == 'Mated Pair #2':
            Q_file = "example_shoeprints/mated_2_q.tiff"
            K_file = "example_shoeprints/mated_2_k.tiff"
            q_border_width = 160
            k_border_width = 160
        elif pair == 'Non-Mated Pair #1':
            Q_file = "example_shoeprints/nonmated_1_q.tiff"
            K_file = "example_shoeprints/nonmated_1_k.tiff"
            q_border_width = 160
            k_border_width = 160
        elif pair == 'Non-Mated Pair #2':
            Q_file = "example_shoeprints/nonmated_2_q.tiff"
            K_file = "example_shoeprints/nonmated_2_k.tiff"
            q_border_width = 160
            k_border_width = 160

    col1, col2 = st.columns([1, 1.5])
    with col1:
        _, c2, _ = st.columns([1, 3, 1])
        with c2:
            st.image("logo.png")
    with col2:
        st.title("SoleMate")
        st.subheader("An End-To-End System for Shoeprint Pattern Matching")
        st.markdown("Williams College SMALL REU 2023")

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
                run the algorithm and see the results!")
    
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
                    new_q_pct = sc.percent_overlap()
                    new_k_pct = sc.percent_overlap(Q_as_base=False)
                    dist_metrics = sc.min_dist()
                    all_cluster_metrics = sc.cluster_metrics()
                    # Subsetting only cluster metrics with n_clusters=20
                    cluster_metrics = {}
                    cluster_metrics['centroid_distance'] = all_cluster_metrics['centroid_distance_n_clusters_20']
                    cluster_metrics['cluster_proprtion'] = all_cluster_metrics['cluster_proprtion_n_clusters_20']
                    cluster_metrics['iterations_k'] = all_cluster_metrics['iterations_k_n_clusters_20']
                    cluster_metrics['wcv'] = all_cluster_metrics['wcv_ratio_n_clusters_20']


                    st.header("Metrics")
                    st.markdown(
                        "We quantify similarity using a number of metrics.")

                    # Overlap
                    st.subheader("Overlap")

                    col1, col2 = st.columns(2)
                    with col1:
                        q_pct = plt.figure(figsize=(10, 7))
                        sns.kdeplot(data=load_train(), x="q_pct", hue="mated",
                                    fill=True, palette=[WILLIAMS_PURPLE, WILLIAMS_GOLD], alpha=0.6
                                    )
                        plt.axvline(x=new_q_pct, color='#BD783A',
                                    linestyle='--', linewidth=3)
                        plt.text(new_q_pct, -0.25, 'Test Pair', verticalalignment='bottom',
                                horizontalalignment='center', fontsize=14, weight='bold', color="#BD783A")
                        plt.title("Q Percent Overlap")
                        st.pyplot(q_pct)
                    with col2:
                        k_pct = plt.figure(figsize=(10, 7))
                        sns.kdeplot(data=load_train(), x="k_pct", hue="mated",
                                    fill=True, palette=[WILLIAMS_PURPLE, WILLIAMS_GOLD], alpha=0.6
                                    )
                        plt.axvline(x=new_k_pct, color='#BD783A',
                                    linestyle='--', linewidth=3)
                        plt.text(new_k_pct, -0.25, 'Test Pair', verticalalignment='bottom',
                                horizontalalignment='center', fontsize=14, weight='bold', color="#BD783A")
                        plt.title("K Percent Overlap")
                        st.pyplot(k_pct)

                    with st.expander(":bar_chart: About the metric: Overlap"):
                        st.subheader("Percent Overlap")
                        st.markdown("Percent overlap is the proportion of points in one\
                                    shoeprint that are within three pixels of the other\
                                    shoeprint after alignment. We observe the overlap\
                                    in both directions— that is, K on Q and Q on K. A\
                                    high percent overlap indicates a higher likelihood\
                                    of the shoeprints originating from a mated pair.")

                    # Distance
                    st.subheader("Closest Point Distances")

                    col1, col2 = st.columns(2)
                    with col1:
                        mean = plt.figure(figsize=(10, 7))
                        sns.kdeplot(data=load_train(), x="mean", hue="mated",
                                    fill=True, palette=[WILLIAMS_PURPLE, WILLIAMS_GOLD], alpha=0.6
                                    )
                        plt.axvline(x=dist_metrics['mean'], color='#BD783A',
                                    linestyle='--', linewidth=3)
                        plt.text(dist_metrics['mean'], -0.015, 'Test Pair', verticalalignment='bottom',
                                horizontalalignment='center', fontsize=14, weight='bold', color="#BD783A")
                        plt.title("Mean CP Distance")
                        st.pyplot(mean)
                    with col2:
                        std = plt.figure(figsize=(10, 7))
                        sns.kdeplot(data=load_train(), x="std", hue="mated",
                                    fill=True, palette=[WILLIAMS_PURPLE, WILLIAMS_GOLD], alpha=0.6
                                    )
                        plt.axvline(x=dist_metrics['std'], color='#BD783A',
                                    linestyle='--', linewidth=3)
                        plt.text(dist_metrics['std'], -0.0075, 'Test Pair', verticalalignment='bottom',
                                horizontalalignment='center', fontsize=14, weight='bold', color="#BD783A")
                        plt.title("Standard Deviation CP Distance")
                        st.pyplot(std)

                    col1, col2 = st.columns(2)
                    with col1:
                        p10 = plt.figure(figsize=(10, 7))
                        sns.kdeplot(data=load_train(), x="0.1", hue="mated",
                                    fill=True, palette=[WILLIAMS_PURPLE, WILLIAMS_GOLD], alpha=0.6
                                    )
                        plt.axvline(x=dist_metrics['0.1'], color='#BD783A',
                                    linestyle='--', linewidth=3)
                        plt.text(dist_metrics['0.1'], -0.15, 'Test Pair', verticalalignment='bottom',
                                horizontalalignment='center', fontsize=14, weight='bold', color="#BD783A")
                        plt.title("10th Percentile CP Distance")
                        st.pyplot(p10)
                    with col2:
                        p25 = plt.figure(figsize=(10, 7))
                        sns.kdeplot(data=load_train(), x="0.25", hue="mated",
                                    fill=True, palette=[WILLIAMS_PURPLE, WILLIAMS_GOLD], alpha=0.6
                                    )
                        plt.axvline(x=dist_metrics['0.25'], color='#BD783A',
                                    linestyle='--', linewidth=3)
                        plt.text(dist_metrics['0.25'], -0.075, 'Test Pair', verticalalignment='bottom',
                                horizontalalignment='center', fontsize=14, weight='bold', color="#BD783A")
                        plt.title("25th Percentile CP Distance")
                        st.pyplot(p25)

                    col1, col2 = st.columns(2)
                    with col1:
                        p50 = plt.figure(figsize=(10, 7))
                        sns.kdeplot(data=load_train(), x="0.5", hue="mated",
                                    fill=True, palette=[WILLIAMS_PURPLE, WILLIAMS_GOLD], alpha=0.6
                                    )
                        plt.axvline(x=dist_metrics['0.5'], color='#BD783A',
                                    linestyle='--', linewidth=3)
                        plt.text(dist_metrics['0.5'], -0.025, 'Test Pair', verticalalignment='bottom',
                                horizontalalignment='center', fontsize=14, weight='bold', color="#BD783A")
                        plt.title("Median CP Distance")
                        st.pyplot(p50)
                    with col2:
                        p75 = plt.figure(figsize=(10, 7))
                        sns.kdeplot(data=load_train(), x="0.75", hue="mated",
                                    fill=True, palette=[WILLIAMS_PURPLE, WILLIAMS_GOLD], alpha=0.6
                                    )
                        plt.axvline(x=dist_metrics['0.75'], color='#BD783A',
                                    linestyle='--', linewidth=3)
                        plt.text(dist_metrics['0.75'], -0.01, 'Test Pair', verticalalignment='bottom',
                                horizontalalignment='center', fontsize=14, weight='bold', color="#BD783A")
                        plt.title("75th Percentile CP Distance")
                        st.pyplot(p75)

                    __, col2, __ = st.columns([1, 2, 1])
                    with col2:
                        p90 = plt.figure(figsize=(10, 7))
                        sns.kdeplot(data=load_train(), x="0.9", hue="mated",
                                    fill=True, palette=[WILLIAMS_PURPLE, WILLIAMS_GOLD], alpha=0.6
                                    )
                        plt.axvline(x=dist_metrics['0.9'], color='#BD783A',
                                    linestyle='--', linewidth=3)
                        plt.text(dist_metrics['0.9'], -0.005, 'Test Pair', verticalalignment='bottom',
                                horizontalalignment='center', fontsize=14, weight='bold', color="#BD783A")
                        plt.title("90th Percentile CP Distance")
                        st.pyplot(p90)

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
                        sns.kdeplot(data=load_train(), x="centroid_distance", hue="mated",
                                    fill=True, palette=[WILLIAMS_PURPLE, WILLIAMS_GOLD], alpha=0.6
                                    )
                        plt.axvline(x=cluster_metrics['centroid_distance'], color='#BD783A',
                                    linestyle='--', linewidth=3)
                        plt.text(cluster_metrics['centroid_distance'], -0.001, 'Test Pair', verticalalignment='bottom',
                                horizontalalignment='center', fontsize=14, weight='bold', color="#BD783A")
                        plt.title("Centroid Distance")
                        st.pyplot(centroid_distance)
                    with col2:
                        cluster_proprtion = plt.figure(figsize=(10, 7))
                        sns.kdeplot(data=load_train(), x="cluster_proprtion", hue="mated",
                                    fill=True, palette=[WILLIAMS_PURPLE, WILLIAMS_GOLD], alpha=0.6
                                    )
                        plt.axvline(x=cluster_metrics['cluster_proprtion'], color='#BD783A',
                                    linestyle='--', linewidth=3)
                        plt.text(cluster_metrics['cluster_proprtion'], -5, 'Test Pair', verticalalignment='bottom',
                                horizontalalignment='center', fontsize=14, weight='bold', color="#BD783A")
                        plt.title("Cluster Proportion")
                        st.pyplot(cluster_proprtion)

                    col1, col2 = st.columns(2)
                    with col1:
                        iterations_k = plt.figure(figsize=(10, 7))
                        sns.kdeplot(data=load_train(), x="iterations_k", hue="mated",
                                    fill=True, palette=[WILLIAMS_PURPLE, WILLIAMS_GOLD], alpha=0.6
                                    )
                        plt.axvline(x=cluster_metrics['iterations_k'], color='#BD783A',
                                    linestyle='--', linewidth=3)
                        plt.text(cluster_metrics['iterations_k'], -0.004, 'Test Pair', verticalalignment='bottom',
                                horizontalalignment='center', fontsize=14, weight='bold', color="#BD783A")
                        plt.title("Iterations K")
                        st.pyplot(iterations_k)
                    with col2:
                        wcv = plt.figure(figsize=(10, 7))
                        sns.kdeplot(data=load_train(), x="wcv", hue="mated",
                                    fill=True, palette=[WILLIAMS_PURPLE, WILLIAMS_GOLD], alpha=0.6
                                    )
                        plt.axvline(x=cluster_metrics['wcv'], color='#BD783A',
                                    linestyle='--', linewidth=3)
                        plt.text(cluster_metrics['wcv'], -0.25, 'Test Pair', verticalalignment='bottom',
                                horizontalalignment='center', fontsize=14, weight='bold', color="#BD783A")
                        plt.title("Within Cluster Variation")
                        st.pyplot(wcv)

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
                                    clusters in K, the difference between the within\
                                    cluster variation in Q and K scaled by the within\
                                    cluster variation in Q, and the number of\
                                    iterations k-means clustering took to find clusters\
                                    in K.")

                st.divider()

                st.header("Classification")
                st.markdown(
                    "Based on a trained random forest model, we hypothesize whether \
                        the input shoeprint pair is mated or non-mated.")

                # Creating a Dataframe row containing the new metrics
                row = {'q_pct': new_q_pct,
                    'k_pct': new_k_pct}
                row.update(dist_metrics)
                row.update(cluster_metrics)
                row = pd.DataFrame(row, index=[0])
                
                # Loading cached model
                rf_model = load_model()
                # Subsetting relevant features
                row = row[list(rf_model.feature_names_in_)].round(2)
                # Indexing into rf probability for class=1 i.e. mated=True
                score = rf_model.predict_proba(row)[0][1]
                
                prob = round(score, 2)
                mated = "Mated" if score > 0.5 else "Non-Mated"
                st.success(f"Our model predicts that the shoeprints are **_{mated}_**", icon="👟")
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
                    
                with st.expander(":technologist: Our random forest implentation"):
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
                                image appears in both the training and test set).")
    
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
    st.set_page_config(layout="wide")
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
