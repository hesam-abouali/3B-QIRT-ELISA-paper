"""Cache of mRMR rankings for the permutation test.

Within a label assignment, the two recipes (mRMR + LogReg_L2 and mRMR + SVM_linear) fit mRMR on the same
training data in every inner and outer fold. install() wraps nested_cv.MRMRSelector.fit so that a fit
with inputs identical to an earlier one (same feature values, labels, max_features and random_state)
reuses its scores and ranking instead of recomputing them. The rankings, and therefore all results, are
the same as without the cache. Only fits with an integer random_state are cached.
"""

import numbers


def install(nested_cv, maxsize=4096):
    if not isinstance(maxsize, numbers.Integral) or maxsize < 1:
        raise ValueError("maxsize must be a positive integer")
    installed = getattr(nested_cv.MRMRSelector.fit, "mrmr_cache", None)
    if installed is not None:  # already installed: keep the existing cache
        return installed
    uncached_fit = nested_cv.MRMRSelector.fit
    cache = {}
    stats = {"calls": 0, "hits": 0}

    def fit(self, X, y, X_val=None, y_val=None):
        stats["calls"] += 1
        # Only fits with an integer seed are deterministic, so only those are cached.
        if (X_val is not None or y_val is not None
                or not isinstance(self.random_state, numbers.Integral)):
            return uncached_fit(self, X, y, X_val, y_val)
        key = (X.dtype.str, X.shape, X.tobytes(), tuple(y.tolist()), self.max_features, self.random_state)
        if key in cache:
            stats["hits"] += 1
            self.scores_, self.ranking_ = tuple(a.copy() for a in cache[key])
            return self
        result = uncached_fit(self, X, y)
        if len(cache) >= maxsize:
            cache.pop(next(iter(cache)))
        cache[key] = (self.scores_.copy(), self.ranking_.copy())
        return result

    fit.mrmr_cache = (stats, cache)
    nested_cv.MRMRSelector.fit = fit
    return stats, cache
