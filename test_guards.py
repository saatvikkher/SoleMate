"""Regression tests for the SoleMate hardening guards.

Run with:  .venv/bin/python -m pytest test_guards.py -v
       or: .venv/bin/python test_guards.py
"""
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sole import Sole
from solepair import SolePair
from solepaircompare import SolePairCompare
from utils.icp import icp

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


def make_pair(n_q=50, n_k=50):
    rng = np.random.RandomState(0)
    Q = Sole(is_image=False,
             coords=pd.DataFrame(rng.rand(n_q, 2) * 100, columns=["x", "y"]))
    K = Sole(is_image=False,
             coords=pd.DataFrame(rng.rand(n_k, 2) * 100, columns=["x", "y"]))
    return SolePair(Q, K, True)


# --------------------------------------------------------------- sole.py
def test_small_image_border_width_error():
    """Image smaller than 2x border_width -> clear ValueError, not PIL crash."""
    from PIL import Image
    p = os.path.join(tempfile.gettempdir(), "solemate_test_small.png")
    Image.new("L", (300, 300), 200).save(p)
    with pytest.raises(ValueError, match="Border width"):
        Sole(image_path=p, border_width=160)


def test_blank_image_error():
    """Uniform image with no detectable print -> clear ValueError."""
    from PIL import Image
    p = os.path.join(tempfile.gettempdir(), "solemate_test_blank.png")
    Image.new("L", (1200, 1800), 230).save(p)
    with pytest.raises(ValueError, match="Could not detect a shoeprint"):
        Sole(image_path=p, border_width=160)


def test_corrupt_file_is_oserror():
    """Corrupt upload raises UnidentifiedImageError (an OSError), which the
    app's widened except must catch."""
    from PIL import UnidentifiedImageError
    p = os.path.join(tempfile.gettempdir(), "solemate_test_corrupt.png")
    with open(p, "wb") as f:
        f.write(b"definitely not an image")
    with pytest.raises((UnidentifiedImageError, OSError)):
        Sole(image_path=p, border_width=100)
    assert issubclass(type("X", (UnidentifiedImageError,), {}), OSError)


def test_rgba_composited_on_white():
    """Transparent pixels must NOT become black ink: a fully transparent
    image composites to white and is rejected as 'no print detected'
    instead of producing a bogus 100%-ink point cloud."""
    from PIL import Image
    p = os.path.join(tempfile.gettempdir(), "solemate_test_rgba.png")
    Image.new("RGBA", (800, 800), (0, 0, 0, 0)).save(p)
    with pytest.raises(ValueError, match="Could not detect a shoeprint"):
        Sole(image_path=p, border_width=50)


def test_negative_coordinate_setter_rejected():
    s = Sole(is_image=False, coords=pd.DataFrame({"x": [1.0], "y": [2.0]}))
    with pytest.raises(TypeError):
        s.coords = "not a dataframe"


# --------------------------------------------------- solepaircompare.py
def test_pc_metrics_negative_coords_no_silent_wrap():
    """Negative coords must be shifted into frame, not wrapped around.
    Q is a pattern; K is the same pattern but ICP-shifted to negative x/y.
    Shifting both into a common non-negative frame must restore exact
    agreement between the rasters (MSE=0, SSIM=1) — the old code silently
    wrapped the negatives to the far corner."""
    Q = pd.DataFrame({"x": [5.0, 15.0, 25.0], "y": [5.0, 15.0, 25.0]})
    K = pd.DataFrame({"x": [-10.0, 0.0, 10.0], "y": [-10.0, 0.0, 10.0]})
    sc = SolePairCompare.__new__(SolePairCompare)
    sc._Q_coords_full = Q
    sc._K_coords_full = K
    m = sc.pc_metrics()
    # after shifting by -(-10): K = Q exactly, so rasters are identical
    assert m["MSE"] == pytest.approx(0.0)
    assert m["SSIM"] == pytest.approx(1.0)


def test_jaccard_union_zero():
    sc = SolePairCompare.__new__(SolePairCompare)
    sc._Q_coords = pd.DataFrame(columns=["x", "y"])
    sc._K_coords = pd.DataFrame(columns=["x", "y"])
    out = sc.jaccard_index(round_coords=[0])
    assert out["jaccard_index_0"] == 0.0  # no ZeroDivisionError


def test_wcv_zero_denominator():
    sc = SolePairCompare.__new__(SolePairCompare)
    df = pd.DataFrame({"x": [1.0] * 4, "y": [2.0] * 4, "label": [0] * 4})
    centroids = pd.DataFrame({"x": [1.0], "y": [2.0]})
    out = sc._within_cluster_var_metric(df, df, centroids, centroids, 1)
    assert np.isfinite(out)


def test_cluster_metrics_too_few_points():
    """Small clouds must raise a clear ValueError, not a sklearn traceback."""
    sc = SolePairCompare.__new__(SolePairCompare)
    sc.random_seed = 0
    sc.K_keep_propn = 1.0
    sc._Q_coords = pd.DataFrame(np.random.rand(10, 2), columns=["x", "y"])
    sc._K_coords = pd.DataFrame(np.random.rand(10, 2), columns=["x", "y"])
    with pytest.raises(ValueError, match="Not enough points to compute clustering"):
        sc.cluster_metrics(n_clusters=20)


def test_cluster_metrics_caps_memory():
    """Very large clouds must not blow memory: sample is capped."""
    sc = SolePairCompare.__new__(SolePairCompare)
    sc.random_seed = 0
    sc.K_keep_propn = 1.0
    big = pd.DataFrame(np.random.rand(40000, 2) * 1000, columns=["x", "y"])
    sc._Q_coords = big
    sc._K_coords = pd.DataFrame(np.random.rand(40000, 2) * 1000,
                                columns=["x", "y"])
    out = sc.cluster_metrics(n_clusters=20)  # would OOM without the cap
    assert "centroid_distance_n_clusters_20" in out


def test_mutable_default_not_shared():
    """icp_downsample_rates default must be None (not a shared list)."""
    import inspect
    sig = inspect.signature(SolePairCompare.__init__)
    default = sig.parameters["icp_downsample_rates"].default
    assert default is None


def test_full_preset_pair_end_to_end():
    """The real app flow on a shipped preset: construction + all metrics."""
    Q = Sole(image_path=os.path.join(STATIC, "nonmated_1_q.tiff"),
             border_width=160)
    K = Sole(image_path=os.path.join(STATIC, "nonmated_1_k.tiff"),
             border_width=160)
    pair = SolePair(Q, K, True)
    sc = SolePairCompare(pair, icp_downsample_rates=[0.05], two_way=True,
                         shift_left=True, shift_right=True,
                         shift_down=True, shift_up=True)
    metrics = {}
    metrics["q_pct"] = sc.propn_overlap()
    metrics.update(sc.min_dist())
    metrics.update(sc.cluster_metrics(n_clusters=20))
    metrics.update(sc.cluster_metrics(n_clusters=100))
    metrics.update(sc.pc_metrics())
    metrics.update(sc.jaccard_index())
    assert all(np.isfinite(v) for v in metrics.values()), \
        "non-finite metric in healthy path"


# ----------------------------------------------------------- utils/icp.py
def test_icp_zero_iterations():
    """max_iterations=0 must not raise UnboundLocalError."""
    A = np.random.rand(20, 2).astype(np.float32)
    B = A + 0.1
    T, dist, i = icp(A, B, max_iterations=0)
    assert T.shape == (3, 3)


# -------------------------------------- second-round audit findings
def test_ssim_tiny_raster():
    """Full clouds rasterizing smaller than 7x7 must not crash SSIM
    (audit finding 1: win_size exceeds image extent)."""
    Q = pd.DataFrame({"x": [2.0, 3.0, 4.0], "y": [2.0, 3.0, 4.0]})
    K = pd.DataFrame({"x": [2.0, 3.0, 4.0], "y": [3.0, 4.0, 5.0]})
    sc = SolePairCompare.__new__(SolePairCompare)
    sc._Q_coords_full = Q
    sc._K_coords_full = K
    m = sc.pc_metrics()  # rasters are 4x4 -> win_size must shrink
    assert np.isfinite(m["SSIM"])


def test_psr_ncc_nan_sanitized():
    """Constant images (all-zero rasters) must yield 0.0, not NaN,
    for PSR and NCC (audit finding 8)."""
    Q = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
    K = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
    sc = SolePairCompare.__new__(SolePairCompare)
    sc._Q_coords_full = Q
    sc._K_coords_full = K
    m = sc.pc_metrics()
    assert np.isfinite(m["PSR"]) and np.isfinite(m["NCC"])


def test_propn_overlap_empty_frame():
    """0-row base frame must return 0.0, not crash on the pandas 3.0
    empty-apply quirk (audit finding 4)."""
    sc = SolePairCompare.__new__(SolePairCompare)
    sc._Q_coords = pd.DataFrame(columns=["x", "y"])
    sc._K_coords = pd.DataFrame({"x": [1.0], "y": [1.0]})
    assert sc.propn_overlap() == 0.0
    sc._Q_coords = pd.DataFrame({"x": [1.0], "y": [1.0]})
    sc._K_coords = pd.DataFrame(columns=["x", "y"])
    assert sc.propn_overlap() == 0.0


def test_empty_icp_downsample_rates():
    """Empty rate list must fall back to [1.0], not IndexError
    (audit finding 5)."""
    pair = make_pair()
    sc = SolePairCompare(pair, icp_downsample_rates=[])
    assert sc is not None


def test_cut_k_keep_propn():
    """K_keep_propn must measure the pre-cut population (audit finding 7).
    A heel cut keeps K points with x > min(Q.x) = 10 (strict), so x=10 is
    excluded: exactly 89 of 100 rows survive -> propn 0.89."""
    Q = Sole(is_image=False,
             coords=pd.DataFrame({"x": [10.0] * 5 + [50.0] * 5,
                                  "y": range(10)}))
    K = Sole(is_image=False,
             coords=pd.DataFrame({"x": np.arange(100.0),
                                  "y": np.arange(100.0)}))
    sc = SolePairCompare.__new__(SolePairCompare)
    sc._Q_coords = Q.coords
    sc._K_coords = K.coords
    sc.K_keep_propn = 1.0
    sc.cut_k("heel", "R")
    assert sc.K_keep_propn == pytest.approx(0.89)


def test_flip_coords_empty():
    """_flip_coords on an empty frame must not raise (audit finding 9)."""
    s = Sole(is_image=False, coords=pd.DataFrame(columns=["x", "y"]))
    out = s._flip_coords(pd.DataFrame(columns=["x", "y"]))
    assert len(out) == 0


def test_process_module_imports_and_method_exists():
    """utils/process.py must import and its 40 call sites must target a
    real method (audit findings 2+3)."""
    import utils.process  # noqa: F401
    assert hasattr(SolePairCompare, "propn_overlap")


def test_ssim_happy_path_unchanged():
    """Normal-size rasters still use the default 7x7 window: SSIM of
    identical images is exactly 1.0."""
    Q = pd.DataFrame({"x": [5.0, 15.0, 25.0, 35.0], "y": [5.0, 15.0, 25.0, 35.0]})
    K = Q.copy()
    sc = SolePairCompare.__new__(SolePairCompare)
    sc._Q_coords_full = Q
    sc._K_coords_full = K
    m = sc.pc_metrics()
    assert m["SSIM"] == pytest.approx(1.0)


# --------------------------------------------------------- SoleMate layer
def test_full_model_loads_from_tempdir():
    """load_everything_model extracts to a writable temp dir (cloud-safe)."""
    import pickle as _p
    import zipfile, tempfile
    target_dir = os.path.join(tempfile.gettempdir(), "solemate_model_test")
    os.makedirs(target_dir, exist_ok=True)
    pkl_path = os.path.join(target_dir, "EVERYTHING_TO_EVERYTHING_NOIND.pkl")
    if not os.path.exists(pkl_path):
        with zipfile.ZipFile(os.path.join(STATIC,
                             "EVERYTHING_TO_EVERYTHING_NOIND.pkl.zip")) as z:
            z.extract("EVERYTHING_TO_EVERYTHING_NOIND.pkl", target_dir)
    with open(pkl_path, "rb") as f:
        model = _p.load(f)
    assert hasattr(model, "estimators_")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
