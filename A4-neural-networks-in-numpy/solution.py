"""
Instructor-only answer key for A4 answers.py.

NOT shipped in the student bundle — exclude alongside grader/grader_impl.py.
"""

# ---------------------------------------------------------------------------
# Task J - Training and the Learning Rate
# ---------------------------------------------------------------------------

j_best_lr: float = 1.0
j_lr_001_underfits: bool = True
j_lr_10_unstable: bool = True


# ---------------------------------------------------------------------------
# Task K - Decision Boundaries
# ---------------------------------------------------------------------------

k_observations: dict[str, bool] = {
    # A linear model (no hidden layer) can separate linearly separable data
    # but XOR is not linearly separable → True by definition.
    "linear_model_solves_linear_but_not_xor": True,
    # Linear model on moons: acc=0.90. Boundary is a straight diagonal line;
    # 0.90 is borderline — crescents are roughly separable along one axis at
    # this sample size but the shape is not truly linearly separable.
    "linear_model_separates_moons": False,  # acc=0.90 is borderline # noqa: E501
    # Stacking linear layers without a non-linearity is still a linear map
    # → boundary is still a hyperplane → True by definition.
    "stacked_linear_no_activation_stays_straight": True,
    # ReLU is piecewise-linear, so its decision boundary is composed of
    # linear segments → True by definition.
    "relu_boundary_is_piecewise_linear": True,
    # [4]+ReLU on circles (acc=0.98): boundary visually encloses the inner ring
    "relu_encloses_inner_circle": True,
    # [4]+ReLU on checkerboard (acc=0.60): clearly below 0.75 → underfits
    "small_relu_underfits_checkerboard": True,
    # XOR accuracies differ across rows (0.65/0.68/0.55/0.91/0.97) → False
    "all_configs_equal_accuracy_on_xor": False,
    # [64,64]+ReLU on spiral (acc=0.97): complex boundary traces both arms
    "large_relu_traces_spiral_arms": True,
}

# Row-1 panel titles: linear=0.95, moons=0.90, circles=0.45, xor=0.65,
# checkerboard=0.49, spiral=0.57.  Only linear (0.95) strictly exceeds 0.90.
k_num_datasets_linear_solves: int = 1

# Checkerboard accs by row: [4]=0.60, [16]=0.70, [64,64]=0.81.
# First row exceeding 0.75 is [64,64] (0.81).
k_smallest_arch_solving_checkerboard: str = "[64,64]"


# ---------------------------------------------------------------------------
# Task L - Training Diagnostics
# ---------------------------------------------------------------------------

l_observations: dict[str, bool] = {
    # Plot dense — healthy update magnitude ~10^-2 range appears in band.
    "healthy_update_magnitude_lands_between_1e-3_and_1e-2": True,  # dense plot  # noqa: E501
    # Plot dense — healthy grad/weight ratio row shows blue near 10^-2.
    "healthy_grad_weight_ratio_near_001_or_01": True,  # noqa: E501
    # Weight-norm row: MNIST panel shows one line (lr_too_high) shooting up
    # dramatically while others are flat.
    "lr_too_high_weight_norm_explodes_upward": True,
    # Update-magnitude row: lr_too_high line is NOT highest after initial
    # spike (weight explosion deflates per-weight update) → False
    "a_higher_lr_always_produces_a_higher_update_magnitude_in_a_useful_range": False,  # noqa: E501
    # Update-magnitude row: lr_too_low line is clearly the lowest, ~10^-4,
    # orders of magnitude below healthy ~10^-2.
    "lr_too_low_update_magnitude_is_far_below_healthy": True,
    # Gradient-norm row: lr_too_low stays essentially flat/near-zero.
    "lr_too_low_gradient_norm_stays_pinned_at_initial_value": True,  # noqa: E501
    # Weight-norm row: deep_bad_init starts at a visibly lower value than
    # healthy in both panels.
    "deep_bad_init_weight_norm_starts_much_lower_than_healthy": True,  # noqa: E501
    # Weight-norm row: lines are clearly at different levels → False
    "all_conditions_converge_to_the_same_weight_norm": False,
    # Update-magnitude row: deep_bad_init line tracks the healthy line closely.
    "deep_bad_init_aggregate_update_magnitude_tracks_healthy_despite_vanishing": True,  # noqa: E501
}

# Gradient-norm row: lr_too_high line is highest in both panels (MNIST panel
# shows it peaking above all others early on).
l_condition_with_highest_peak_grad_norm: str = "lr_too_high"

# Update-magnitude row: lr_too_low is clearly the lowest line (~10^-4).
l_condition_with_lowest_update_magnitude: str = "lr_too_low"

# Gradient-norm row: deep_bad_init and lr_too_low are both low; the plot is
# dense enough that the ordering between them is uncertain.
l_condition_with_lowest_gradient_norm: str = "deep_bad_init"  # noqa: E501


# ---------------------------------------------------------------------------
# Task M - Regression
# ---------------------------------------------------------------------------

m_observations: dict[str, bool] = {
    # Regression has a single linear output neuron — no softmax by design
    "regression_uses_single_output_no_softmax": True,
    # [1] hidden unit cannot capture non-linear shape → True by definition
    "underfit_model_misses_curve_shape": True,
    # [8,8] predicted curve closely tracks the true dashed curve
    "mid_model_tracks_true_curve_best": True,
    # Test MSE: [1]=0.1856, [8,8]=0.0177, [64,64,64]=0.0470 → not monotone
    "more_hidden_units_always_lower_test_mse": False,
    # [64,64,64] panel clearly shows a wiggly orange curve chasing noise
    "overfit_model_wiggles_through_noisy_points": True,
    # All three panels show the predicted curve not passing through every point
    "every_model_passes_through_all_noisy_points": False,
    # Panel titles: [64,64,64] train=0.0106 (lowest), test=0.0470 vs
    # [8,8] test=0.0177 → overfit model has higher test MSE than mid model
    "overfit_has_lowest_train_but_higher_test_mse": True,
}

# Panel titles: [8,8] test=0.0177 < [64,64,64] test=0.0470 < [1] test=0.1856
m_best_generalizing_config: str = "[8,8]"

# Panel titles: [64,64,64] train=0.0106 < [8,8] train=0.0197 < [1] train=0.1245
m_lowest_train_mse_config: str = "[64,64,64]"


# ---------------------------------------------------------------------------
# Task N - Input Scaling
# ---------------------------------------------------------------------------

n_observations: dict[str, bool] = {
    # ToArray divides uint8 [0,255] by 255 → maps to [0,1] by definition
    "toarray_maps_pixels_to_unit_interval": True,
    # Normalize subtracts mean and divides by std → zero mean / unit std
    "normalize_centers_to_zero_mean_unit_std": True,
    # Transforms operate on inputs X only, not on integer class labels y
    "scaling_changes_inputs_not_labels": True,
    # Val-accuracy row: raw column — only lr=0.001 (blue) rises, all other
    # lrs stay near 0.1 (diverged or stuck) → True
    "raw_only_trains_at_the_smallest_lr": True,
    # distractor: raw inputs fail to train at larger lrs
    "raw_and_scaled_train_equally_well_at_every_lr": False,
    # Val-accuracy row: toarray and normalized columns show 2+ lrs rising
    # successfully → wider stable range than raw → True
    "scaled_inputs_train_across_a_wider_lr_range": True,
    # distractor: very large lr still destabilizes even after scaling
    "a_larger_lr_is_always_better_after_scaling": False,
    # Grad-norm row (log scale): raw column lines cluster ~10^1–10^2 while
    # toarray and normalized cluster ~10^0–10^1 → raw is larger → True
    "raw_inputs_produce_larger_gradient_norm_than_scaled": True,
    # distractor: update magnitudes differ across pipelines
    "update_magnitude_is_identical_across_all_three_pipelines": False,
    # Update-magnitude row: raw column only lr=0.001 lands in the shaded
    # healthy band; toarray/normalized have 2+ lrs in the band → True
    "scaling_keeps_update_magnitude_in_the_healthy_band_at_more_lrs": True,
}

# Val-accuracy row, raw column: only lr=0.001 rises → best lr for raw = 0.001
n_best_lr_for_raw: float = 0.001

# Val-accuracy row: normalized column achieves the highest val accuracy
# across lrs (multiple curves converge to ~0.88+)
n_best_scaling: str = "normalized"


# ---------------------------------------------------------------------------
# Task O - Hyperparameter Experiments
# ---------------------------------------------------------------------------

o_observations: dict[str, bool] = {
    # Optimizers panel: adam train (green solid) reaches 1.0 by ~epoch 7,
    # sgd train (blue solid) reaches 1.0 by ~epoch 25 → True
    "adam_converges_in_fewer_epochs_than_plain_sgd": True,
    # Optimizers panel: momentum train (orange) reaches 1.0 faster than sgd
    # (blue) → True
    "momentum_speeds_up_plain_sgd": True,
    # Optimizers panel: adam val (green dashed) ~0.94 vs sgd val (blue
    # dashed) ~0.93 at epoch 30 — adam is slightly higher but not
    # guaranteed across all epochs → False
    "adam_always_reaches_higher_final_val_accuracy_than_sgd": False,
    # Regularization panel: none train (blue solid)=1.0, none val~0.89 →
    # gap ~0.11. Dropout train stays ~0.95, val ~0.90 → smaller gap → True
    "unregularized_model_has_the_largest_train_val_gap": True,
    # Regularization panel: dropout train does not reach 1.0 (stays ~0.95),
    # dropout val ~0.90 → smaller train-val gap than none → True
    "dropout_reduces_the_train_val_gap": True,
    # Regularization typically hurts training accuracy
    "regularization_increases_training_accuracy": False,
    # Regularization panel: weight_decay train reaches 1.0 and val ~0.89–0.90;
    # gap is similar to none (~0.10–0.11) — marginally reduces it.
    "weight_decay_reduces_the_train_val_gap": True,  # noqa: E501
}

# Optimizers panel: adam train curve saturates earliest (~epoch 7 vs
# momentum ~epoch 10 vs sgd ~epoch 25)
o_fastest_optimizer: str = "adam"

# Regularization panel: dropout val (orange dashed) ~0.90, weight_decay
# val (green dashed) ~0.89–0.90, none val (blue dashed) ~0.89. Dropout
# appears slightly highest but the margin is very small.
o_best_regularizer_for_val: str = "dropout"  # noqa: E501


# ---------------------------------------------------------------------------
# Task P - Weight Visualization
# ---------------------------------------------------------------------------

p_observations: dict[str, bool] = {
    # MNIST inputs are 28×28 = 784 features; first-layer weight columns are
    # 784-vectors that reshape to 28×28 images → True by definition
    "first_layer_weight_columns_reshape_to_28x28": True,
    # Baseline figure: H=16 and H=64 filters show clear structure, not the
    # uniform-noise appearance of randomly initialized weights → True
    "filters_are_not_pure_random_noise": True,
    # Baseline figure: H=16 and H=64 filters show arc/stroke patterns that
    # resemble digit strokes or templates → True
    "some_filters_resemble_strokes_or_digit_templates": True,
    # distractor: units detect combinations of features, not single classes
    "each_first_layer_unit_detects_exactly_one_digit_class": False,
    # Baseline figure: H=4 has 4 large bold filters each showing a distinct
    # broad stroke (clearly different from each other) → True
    "with_few_units_each_filter_is_more_distinct": True,
    # Baseline figure: H=4 filters are broad blobs; H=16/H=64 are smaller
    # patterns — not progressively sharper digit images → False
    "increasing_units_makes_each_filter_a_sharper_digit": False,
    # Baseline figure: H=64 has 64 small filters; many share similar arc
    # patterns (distributed / more redundant) → True
    "with_many_units_filters_look_more_redundant_or_distributed": True,
}

# MNIST images are 28×28 → filter side is exactly 28 by definition
p_filter_image_side: int = 28

# Baseline figure panel titles: H=4 acc=0.724, H=16 acc=0.949, H=64 acc=0.976
# → monotonically increasing → True
p_more_units_higher_test_accuracy: bool = True
