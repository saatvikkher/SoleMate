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

train = pd.read_csv("result_train_0711.csv")


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
        border_width = st.slider("Select border width:", 0, 300, 160)

        # Select partial_type
        partial_options = ["full", "partial"]
        partial_type = st.selectbox("Select Partial Type:", partial_options)

    col1, col2 = st.columns([1,1.5])
    with col1:
        c1, c2, c3 = st.columns([1,3,1])
        with c2:
            st.image("logo.png")
    with col2:
        st.title("SoleMate")
        st.subheader("An End-To-End System for Shoe-Print Pattern Matching")
        st.markdown("Williams College SMALL REU 2023")

    st.divider()

    # Summary
    st.header("Summary")
    st.markdown("*SoleMate* is a tool for matching footwear outsole impressions. Our algorithm offers a fast\
                and robust method to match the *K* shoe (the known shoe of a suspect)\
                to a *Q* shoe (the questioned shoeprint found at the crime scene). We use Iterative\
                Closest Point (ICP), a point cloud registration algorithm to find the best affine transformation\
                to align two shoeprints. We then calculate metrics")

    if st.sidebar.button("Run SoleMate"):
        st.divider()
        # Check if both images are uploaded
        if Q_file and K_file:
            Q = Sole(Q_file, border_width=border_width)
            K = Sole(K_file, border_width=border_width)
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
                sns.kdeplot(data=train, x="q_pct", hue="mated",
                            fill=True, palette=[WILLIAMS_PURPLE, WILLIAMS_GOLD], alpha=0.6
                            )
                plt.axvline(x=new_q_pct, color='#BD783A', linestyle='--', linewidth=3)
                plt.text(new_q_pct, -0.25, 'Test Pair',verticalalignment='bottom', horizontalalignment='center', fontsize=14, weight='bold', color="#BD783A")
                plt.title("Q Percent-Overlap")
                st.pyplot(q_pct)
            with col2:
                k_pct = plt.figure(figsize=(10, 7))
                sns.kdeplot(data=train, x="k_pct", hue="mated",
                            fill=True, palette=[WILLIAMS_PURPLE, WILLIAMS_GOLD], alpha=0.6
                            )
                plt.axvline(x=new_k_pct, color='#BD783A', linestyle='--', linewidth=3)
                plt.text(new_k_pct, -0.25, 'Test Pair',verticalalignment='bottom', horizontalalignment='center', fontsize=14, weight='bold', color="#BD783A")
                plt.title("K Percent-Overlap")
                st.pyplot(k_pct)

            # Distance
            st.subheader("Closest Point Distances")

            # Clustering
            st.subheader("Clustering")

    st.divider()
    st.markdown(
        "*Developed and Maintained by Simon Angoluan, Divij Jain, Saatvik Kher, Lena Liang, Yufeng Wu, Ashley Zheng*")


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
