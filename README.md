<!-- PROJECT LOGO -->
<div align="center">
  <a href="https://github.com/saatvikkher/SoleMate/blob/main/logo.png">
    <img src="static/logo.png" alt="Logo" width="120" height="120">
  </a>
  <h1 align="center">SoleMate</h1>
</div>

Official implementation of *"Improving and Evaluating Machine Learning Methods for Forensic Shoeprint Matching"*

Try it out at [solemate.streamlit.app](https://solemate.streamlit.app/)

<!-- 
### Deliverables
- Extract edges from a shoeprint
- Novel Iterative Closest Point (ICP) implementation for improved alignment
- Calculate Similarity Metrics to assess alignment
 -->
 
## Getting Started
See `demo.ipynb` for a pipeline walkthrough.

### Prerequisites

Before you begin, make sure you have the following software installed:

- [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/saatvikkher/SoleMate.git
   ```
2. Install required dependencies
   ```sh
   conda create --name solemate --file requirements.txt
   ```
3. Activate the environment
   ```sh
   conda activate solemate
   ```

## Datasets
To grow and evaluate our model, we use data from three sources within CSAFE:
1. Their [public longitudinal shoe outsole impression dataset](https://forensicstats.org/shoeoutsoleimpressionstudy/). These data correspond to our Pristine AN (Baseline), Partial, and Pristine Time 2 and 3 prints.
2. Their [public 2D footware dataset](https://forensicstats.org/2d-footwear-data-set/). These data correspond to our Pristine 150 prints.
3. Their degraded / blurry print dataset, created in ["The effect of image descriptors on the performance of classifiers of footwear outsole image pairs"](https://doi.org/10.1016/j.forsciint.2021.111126). Available upon request to the authors. These data correspond to our Blurry prints.

## Random forest growth
Following our approach, you can reproduce the growth of our model via the script `grow_solemate_model.py`, which generates a random forest trained on the previously extracted similarity metrics found in `results/` and saves the model to `solemate_model.joblib`.

## Similarity Metric Extraction
To extract fresh similarity metrics from a directory of images, see the script `extract-solemate-metrics-pipeline.py` and the function `parseArgs()` for a description of important command line arguments.

## Model Predictions
To get probability of matedness (that both prints came from the same shoe) for pairs, use `get_model_predictions.py` which expects a `solemate_model.joblib` model. This can be run like:
```
python get_model_predictions.py --metrics_fp results/RESULT_BASELINE_TEST.csv --output_fp model_predictions
```

*Developed by Simon Angoluan, Divij Jain, Saatvik Kher, Lena Liang, Myer Liebman, Simon Socolow, Charlie Theras, Yufeng Wu, and Ashley Zheng.*

*We conducted our research in collaboration with the [Center for Statistics and Applications in Forensic Evidence](https://forensicstats.org/).*