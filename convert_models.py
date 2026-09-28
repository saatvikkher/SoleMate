"""One-time migration: upgrade pickled sklearn models to the current sklearn format.

The models shipped with SoleMate were saved with scikit-learn 1.0.2. Old tree
pickles store node arrays with a 7-field dtype; sklearn >= 1.3 requires an
8th field (`missing_go_to_left`) and validates it inside Tree.__setstate__.
Since sklearn 1.0.2 cannot even be installed on Python 3.14, we upgrade the
pickles in place:

  1. Unpickle with a custom Unpickler that maps `sklearn.tree._tree.Tree`
     to a Python subclass whose __setstate__ converts the old node array to
     the current NODE_DTYPE (zero-filling missing_go_to_left — correct, since
     pre-1.3 trees never branched on missing values).
  2. Replace the shim instances with plain Tree objects (so the re-pickled
     file references only the genuine class).
  3. Re-pickle.

Usage:  .venv/bin/python convert_models.py [--backup]
"""
import io
import pickle
import shutil
import sys
import warnings
import zipfile

import numpy as np
import sklearn

from sklearn.tree._tree import Tree  # cython extension type (subclasses fine)


class _TreeShim(Tree):
    """Tree subclass that accepts pre-1.3 (7-field) node arrays."""

    def __setstate__(self, state):
        nodes = state.get('nodes') if isinstance(state, dict) else None
        if nodes is not None and getattr(nodes.dtype, 'names', None) \
                and 'missing_go_to_left' not in nodes.dtype.names:
            state['nodes'] = _upgrade_nodes(nodes)
        try:
            super().__setstate__(state)
        except TypeError:
            # Some versions have no cdef __setstate__ signature quirks; the
            # plain path always works after the dtype is upgraded.
            Tree.__setstate__(self, state)


def _current_node_dtype():
    return Tree(1, np.array([0], dtype=np.intp), 1).__getstate__()['nodes'].dtype


def _upgrade_nodes(nodes):
    """Rebuild an old 7-field node array with the current NODE_DTYPE."""
    cur = _current_node_dtype()
    new_nodes = np.zeros(nodes.shape, dtype=cur)
    for name in nodes.dtype.names:
        new_nodes[name] = nodes[name]
    # missing_go_to_left = 0 everywhere: pre-1.3 trees never branched on
    # missing values, so 0 ("no missing-value branch") is the correct fill.
    return new_nodes


class _UpgradeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'sklearn.tree._tree' and name == 'Tree':
            return _TreeShim
        if module.startswith('numpy.core'):
            # numpy 2.x moved these; old pickles still reference numpy.core.*
            try:
                return super().find_class(module, name)
            except (AttributeError, ModuleNotFoundError):
                return super().find_class('numpy._core' + module[len('numpy'):], name)
        return super().find_class(module, name)


def _fix_shims(obj):
    """Replace _TreeShim instances reachable from obj with plain Tree objects."""
    if isinstance(obj, _TreeShim):
        return _plain_tree(obj)
    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            nv = _fix_shims(v)
            if nv is not v:
                obj[k] = nv
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            nv = _fix_shims(v)
            if nv is not v:
                obj[i] = nv
    elif isinstance(obj, tuple):
        if any(isinstance(v, _TreeShim) for v in obj):
            return tuple(_fix_shims(v) for v in obj)
    elif isinstance(obj, np.ndarray) and obj.dtype == object:
        for idx in np.ndindex(obj.shape):
            v = obj[idx]
            nv = _fix_shims(v)
            if nv is not v:
                obj[idx] = nv
    elif hasattr(obj, '__dict__'):
        for attr, v in vars(obj).copy().items():
            nv = _fix_shims(v)
            if nv is not v:
                setattr(obj, attr, nv)
    return obj


def _plain_tree(shim):
    state = shim.__getstate__()
    tree = Tree(shim.n_features, shim.n_classes, shim.n_outputs)
    tree.__setstate__(state)
    return tree


def _fix_forest_template(obj):
    """Ensure forest objects expose the modern `estimator` template attribute.

    Pickles from sklearn <= 1.1 store the template tree as `base_estimator`;
    sklearn >= 1.2 renamed it to `estimator` (and its tags/fit code accesses
    self.estimator). Unpickling never runs __init__, so we set it here.
    """
    if not hasattr(obj, 'estimators_'):  # not a forest ensemble
        return
    if not hasattr(obj, 'estimator'):
        if hasattr(obj, 'base_estimator'):
            import copy
            obj.estimator = copy.deepcopy(obj.base_estimator)
        else:
            from sklearn.tree import DecisionTreeClassifier
            obj.estimator = DecisionTreeClassifier()
        print(f'   set .estimator from .base_estimator '
              f'({type(obj.estimator).__name__})')


def _normalize_classifier_tree_values(est):
    """Normalize a classifier's tree `value` array in place.

    Old pickles (sklearn <= 1.2) store RAW weighted class counts in
    tree.value and the estimator renormalized at predict time. sklearn >= 1.3
    stores fractions in tree.value (normalization moved into the tree at fit
    time) and no longer renormalizes. Dividing by weighted_n_node_samples
    reproduces exactly what a modern fit would store.
    """
    tree = getattr(est, 'tree_', None)
    if tree is None or not hasattr(est, 'classes_'):
        return  # not a fitted classifier tree
    values = tree.value                      # view into the tree's buffer
    w = tree.weighted_n_node_samples[:values.shape[0]]
    safe_w = np.where(w > 0, w, 1.0)
    values /= safe_w[:, None, None]
    print(f'   normalized tree.value for {type(est).__name__} '
          f'({values.shape[0]} nodes)')


def _postprocess(obj):
    """Recursively fix forests and classifier trees reachable from obj.

    NOTE: each object must be processed exactly ONCE — vars(forest) already
    reaches estimators_, so no explicit second pass over estimators_.
    """
    if isinstance(obj, dict):
        for v in obj.values():
            _postprocess(v)
        return
    if isinstance(obj, (list, tuple, set, frozenset)):
        for v in obj:
            _postprocess(v)
        return
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        for v in obj.ravel():
            _postprocess(v)
        return
    if hasattr(obj, '__dict__'):
        for v in list(vars(obj).values()):
            _postprocess(v)
    if hasattr(obj, 'estimators_'):
        _fix_forest_template(obj)
    if hasattr(obj, 'tree_'):
        _normalize_classifier_tree_values(obj)


def load_old_pickle(raw_bytes):
    return _load_from_bytes(raw_bytes)


def _load_from_bytes(raw_bytes):
    import io
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        unpickler = _UpgradeUnpickler(io.BytesIO(raw_bytes))
        obj = unpickler.load()
    obj = _fix_shims(obj)
    return obj


def convert(src_path, make_backup=False):
    print(f'-- loading {src_path}')
    if src_path.endswith('.zip'):
        with zipfile.ZipFile(src_path) as z:
            member = z.namelist()[0]
            raw = z.read(member)
    else:
        with open(src_path, 'rb') as f:
            raw = f.read()

    obj = _load_from_bytes(raw)
    _postprocess(obj)
    print(f'   unpickled + upgraded trees (sklearn {sklearn.__version__})')

    if make_backup:
        shutil.copy2(src_path, src_path + '.pre-upgrade.bak')
        print(f'   backup: {src_path}.pre-upgrade.bak')

    payload = io.BytesIO()
    pickle.dump(obj, payload, protocol=pickle.HIGHEST_PROTOCOL)

    if src_path.endswith('.zip'):
        with zipfile.ZipFile(src_path, 'w', zipfile.ZIP_DEFLATED) as z:
            z.writestr(member, payload.getvalue())
        print(f'   re-zipped {src_path} ({member}: {len(payload.getvalue())} bytes)')
    else:
        with open(src_path, 'wb') as f:
            f.write(payload.getvalue())
        print(f'   wrote {src_path} ({len(payload.getvalue())} bytes)')


if __name__ == '__main__':
    args = sys.argv[1:]
    make_backup = '--backup' in args
    targets = [a for a in args if not a.startswith('--')] or [
        'static/BASELINE_TO_EVERYTHING.pkl',
        'static/EVERYTHING_TO_EVERYTHING_NOIND.pkl.zip',
    ]
    for t in targets:
        convert(t, make_backup=make_backup)
    print('done.')
