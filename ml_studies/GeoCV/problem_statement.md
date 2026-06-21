# Problem Statement: Spatial Cross-Validation for Random Forests in Python

## The gap

Standard [k-fold cross-validation](k_fold_cross_validation.md) is invalid for spatial data, yet it's the default almost everyone reaches for when running `RandomForestRegressor`/`Classifier` on geospatial features. The reason, precisely: a random forest evaluated with ordinary Cross Validation gives *optimistically biased* accuracy estimates whenever nearby points are spatially autocorrelated ([Tobler's First Law](toblers_first_law.md) — "everything is related to everything else, but near things are more related"), because a random fold split puts training points right next to test points, leaking spatial information. The model isn't actually predicting well in genuinely unseen areas; it's interpolating between training points that happen to be 50 meters from a test point.

This matters in domains like heat demand modeling, building stock characterization, and retrofit potential estimation, all spatially autocorrelated. A model "validated" with standard Cross Validation looks great in-sample and then fails when asked to predict for a genuinely new district.

The fix: **spatial cross-validation** (buffered/blocked spatial CV, per [Roberts et al. 2017](roberts_et_al.md), and [Valavi et al.'s](valavi_et_al.md) `blockCV` in R) — exists in the R ecosystem (`blockCV`, `sperrorest`) and in scattered research code, but **there's no clean, well-maintained, sklearn-compatible Python package that does this properly for random forests specifically**, with good defaults and PostGIS-native input. Python's geospatial-ML tooling (`pyspatialml` etc.) is comparatively immature here.

## What's needed

An sklearn-compatible wrapper that:

1. **Takes a `GeoDataFrame`** (or raw lat/lon + features) and a fitted/unfitted `RandomForestRegressor`/`Classifier`.
2. **Implements proper spatial CV splitters** — at minimum: blocked spatial K-fold (partition space into grid blocks, hold out whole blocks) and buffered leave-one-out (exclude a radius around each test point from training, per Le Rest et al. 2014's recommendation to avoid both leakage *and* the conservative-bias issues blocking alone can introduce).
3. **Diagnoses the autocorrelation problem automatically** — compute Moran's I on the residuals, and warn the user if their standard CV score is likely inflated, with a side-by-side comparison: e.g. "Standard CV R² = 0.91, Spatial CV R² = 0.62 — your model may not generalize to new areas."
4. **Gives spatially-aware feature importance** — standard permutation importance is also biased by autocorrelation (shuffling a feature for one point barely changes the answer if its spatial neighbors are still informative); a spatial-block permutation fixes this the same way.