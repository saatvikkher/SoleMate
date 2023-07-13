import streamlit as st
import pandas as pd
import numpy as np
from process_util import process_image
from sole import Sole
from solepair import SolePair
from solepaircompare import SolePairCompare
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from util import WILLIAMS_GOLD, WILLIAMS_PURPLE
import pickle


@st.cache_data
def load_train():
    train = pd.read_csv("result_train_0711.csv")
    return train


@st.cache_resource
def load_full_model():
    full_model = pickle.load(open('full.pkl', 'rb'))
    return full_model


@st.cache_resource
def load_partial_model():
    partial_model = pickle.load(open('full.pkl', 'rb'))
    return partial_model


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
    # Sidebar
    with st.sidebar:
        # Upload images
        Q_file = st.file_uploader("Upload shoeprint Q", type=["tiff"])
        K_file = st.file_uploader("Upload shoeprint K", type=["tiff"])

        # Select border-width
        q_border_width = st.slider("Select Q border width:", 0, 300, 160)
        k_border_width = st.slider("Select K border width:", 0, 300, 160)

        # Select partial_type
        partial_options = ["full", "partial"]
        partial_type = st.selectbox("Select partial type:", partial_options)

    col1, col2 = st.columns([1, 1.5])
    with col1:
        _, c2, _ = st.columns([1, 3, 1])
        with c2:
            st.image("logo.png")
    with col2:
        st.title("SoleMate")
        st.subheader("An End-To-End System for Shoe-Print Pattern Matching")
        st.markdown("Williams College SMALL REU 2023")

    st.divider()

    # Summary
    st.header("Summary")
    st.markdown("*SoleMate* is a tool for determining whether or not footwear outsole impressions match. Our algorithm offers a fast\
                and robust method to match the *K* shoe (the known shoe of a suspect)\
                to a *Q* shoe (the questioned shoeprint found at the crime scene). We use Iterative\
                Closest Point (ICP), a point cloud registration algorithm to find the best affine transformation\
                to align two shoeprints. We then calculate metrics from which we classify the pair as a mate or\
                non-mate based on the results of a random forest model.")
    st.markdown("To use this tool, upload two images on the left: a Q shoe and\
                a K shoe. If the images have a frame or ruler, designate the\
                width of the border in pixels with the slider. If the Q print\
                is partial, select partial from the dropdown menu to compare\
                the metrics to those from the distributions of partial prints.\
                Finally, click the button to run the algorithm and see the\
                results!")

    if st.sidebar.button("Run SoleMate"):
        st.divider()
        # Check if both images are uploaded
        if Q_file and K_file:
            Q = Sole(Q_file, border_width=q_border_width)
            K = Sole(K_file, border_width=k_border_width)
            pair = SolePair(Q, K, True)

            # ICP
            st.header("ICP Alignment")
            st.markdown(
                "We calculate the best rigid body transformation to align the K shoe to the Q shoe.")
            with st.spinner("Aligning Soles..."):
                sc = SolePairCompare(pair, icp_downsample_rate=0.02, two_way=True, shift_left=True,
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
                            minimize the root mean squared of the distances.\
                            The algorithm iteratively computes distances and\
                            applies transformations until it determines that a\
                            local minimum root mean square has been reached, at\
                            which point the point clouds should be aligned.")

            st.divider()
            # Metrics
            new_q_pct = sc.percent_overlap()
            new_k_pct = sc.percent_overlap(Q_as_base=False)

            st.header("Metrics")
            st.markdown(
                "We attempt to quantify similarity using a number of metrics.")

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
                plt.title("Q Percent-Overlap")
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
                plt.title("K Percent-Overlap")
                st.pyplot(k_pct)

            with st.expander(":bar_chart: About the metric: Overlap"):
                st.subheader("Percent Overlap")
                st.markdown("Percent overlap is the proportion of points in one\
                            shoeprint that are within the circular region of\
                            radius three coordinates of a point on the other\
                            print after alignment. We observe the overlap in\
                            both directions—that is, K on Q and Q on K—and a\
                            high percent overlap indicates a higher likelihood\
                            of the shoeprints originating from a mated pair.")

            # Distance
            st.subheader("Closest Point Distances")

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
                "Based on a trained random forest model, we hypothesize whether the input shoeprint pair is mated or non-mated.")
            
            with st.expander(":question: What is random forest?"):
                st.subheader("Random Forest")
                st.markdown("write")


    st.markdown(
        "*Disclaimer: This tool should be used for research purposes only.*")
    st.markdown(
        "Developed and Maintained by Simon Angoluan, Divij Jain, Saatvik Kher, Lena Liang, Yufeng Wu, Ashley Zheng")
    st.markdown(
        "Please [reach out to us](saatvikkher1@gmail.com) if you run into any issues or have any comments or questions.")


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
