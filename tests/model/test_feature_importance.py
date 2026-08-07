"""Tests for f1opt.model.feature_importance (Iter-137)."""
from __future__ import annotations

import numpy as np
import pytest

from f1opt.data.setup_schema import DEFAULT_SETUP
from f1opt.model.feature_importance import (
    FEATURE_GROUPS,
    FEATURE_NAMES,
    feature_importance_summary,
    gradient_feature_importance,
    permutation_feature_importance,
    rank_features,
)
from f1opt.model.surrogate import INPUT_DIM, SurrogateModel, build_input_vector
from f1opt.model.train import train


# --- layout invariants ------------------------------------------------------ #
def test_feature_names_length_matches_input_dim() -> None:
    assert len(FEATURE_NAMES) == INPUT_DIM == 39


def test_feature_groups_length_and_labels() -> None:
    assert len(FEATURE_GROUPS) == INPUT_DIM
    assert set(FEATURE_GROUPS) == {"setup", "track", "driver"}
    # setup = first 21, track = next 10, driver = last 8.
    assert all(g == "setup" for g in FEATURE_GROUPS[:21])
    assert all(g == "track" for g in FEATURE_GROUPS[21:31])
    assert all(g == "driver" for g in FEATURE_GROUPS[31:])


def test_feature_names_include_known_setup_fields() -> None:
    for field in ("front_wing", "rear_wing", "fuel_load", "front_brake_bias"):
        assert field in FEATURE_NAMES


def test_feature_names_include_track_and_driver_groups() -> None:
    assert "track_length" in FEATURE_NAMES
    assert "track_unknown_flag" in FEATURE_NAMES
    assert "drv_aggression_score" in FEATURE_NAMES
    # one-hot track types present.
    assert "track_type_high_downforce" in FEATURE_NAMES


# --- helpers ---------------------------------------------------------------- #
def _make_batch(n: int = 64, seed: int = 0) -> np.ndarray:
    """Build a (n, 39) batch by perturbing DEFAULT_SETUP across a few tracks."""
    rng = np.random.default_rng(seed)
    tracks = ["silverstone", "monza", "spa", "monaco", "suzuka"]
    rows: list[np.ndarray] = []
    for _ in range(n):
        setup = DEFAULT_SETUP
        # perturb a few fields via from_vector noise.
        vec = np.asarray(setup.to_vector(), dtype=np.float32)
        noise = rng.uniform(-0.05, 0.05, size=vec.shape).astype(np.float32)
        vec = np.clip(vec + noise, 0.0, 1.0)
        from f1opt.data.setup_schema import CarSetup

        setup_p = CarSetup.from_vector(list(vec))
        tid = tracks[rng.integers(len(tracks))]
        rows.append(build_input_vector(setup_p, tid))
    return np.stack(rows, axis=0)


@pytest.fixture(scope="module")
def trained_model() -> SurrogateModel:
    """Small fast model for importance tests."""
    return train(
        iterations=300, n_samples=600, seed=0, log=False, save=False,
        batch_size=32, early_stopping_patience=3,
    )


# --- gradient_feature_importance -------------------------------------------- #
def test_gradient_importance_returns_39_entries(trained_model: SurrogateModel) -> None:
    x = _make_batch(32)
    imp = gradient_feature_importance(trained_model, x)
    assert set(imp.keys()) == set(FEATURE_NAMES)
    assert len(imp) == INPUT_DIM


def test_gradient_importance_non_negative(trained_model: SurrogateModel) -> None:
    x = _make_batch(32)
    imp = gradient_feature_importance(trained_model, x, normalize=False)
    assert all(v >= 0.0 for v in imp.values())


def test_gradient_importance_normalized_sums_to_one(
    trained_model: SurrogateModel,
) -> None:
    x = _make_batch(32)
    imp = gradient_feature_importance(trained_model, x, normalize=True)
    assert abs(sum(imp.values()) - 1.0) < 1e-5


def test_gradient_importance_rejects_bad_shape(trained_model: SurrogateModel) -> None:
    with pytest.raises(ValueError, match="must be"):
        gradient_feature_importance(trained_model, np.zeros((5, 10)))


def test_gradient_importance_deterministic(trained_model: SurrogateModel) -> None:
    x = _make_batch(32)
    a = gradient_feature_importance(trained_model, x, normalize=False)
    b = gradient_feature_importance(trained_model, x, normalize=False)
    assert a == b  # exact — eval mode, no dropout.


# --- permutation_feature_importance ----------------------------------------- #
def test_permutation_importance_returns_39_entries(
    trained_model: SurrogateModel,
) -> None:
    x = _make_batch(32)
    # Pseudo-target: model's own prediction (so baseline MAE ~ 0; shuffling
    # any feature the model actually uses must increase MAE).
    sec, _ = trained_model(torch_from_np(x))
    y = sec.sum(dim=1).detach().cpu().numpy()
    imp = permutation_feature_importance(trained_model, x, y, n_repeats=2, seed=0)
    assert set(imp.keys()) == set(FEATURE_NAMES)


def test_permutation_importance_non_negative(
    trained_model: SurrogateModel,
) -> None:
    x = _make_batch(32)
    sec, _ = trained_model(torch_from_np(x))
    y = sec.sum(dim=1).detach().cpu().numpy()
    imp = permutation_feature_importance(
        trained_model, x, y, n_repeats=2, seed=0, normalize=False
    )
    assert all(v >= 0.0 for v in imp.values())


def test_permutation_importance_normalized_sums_to_one(
    trained_model: SurrogateModel,
) -> None:
    x = _make_batch(32)
    sec, _ = trained_model(torch_from_np(x))
    y = sec.sum(dim=1).detach().cpu().numpy()
    imp = permutation_feature_importance(
        trained_model, x, y, n_repeats=2, seed=0, normalize=True
    )
    # Some features may have 0 importance (sum could be 0 if model ignores all);
    # when non-zero, shares must sum to 1.
    if sum(imp.values()) > 0.0:
        assert abs(sum(imp.values()) - 1.0) < 1e-5


def test_permutation_importance_rejects_bad_y(
    trained_model: SurrogateModel,
) -> None:
    x = _make_batch(32)
    with pytest.raises(ValueError, match="y must be"):
        permutation_feature_importance(
            trained_model, x, np.zeros(10), n_repeats=1
        )


def test_permutation_importance_deterministic_with_seed(
    trained_model: SurrogateModel,
) -> None:
    x = _make_batch(32)
    sec, _ = trained_model(torch_from_np(x))
    y = sec.sum(dim=1).detach().cpu().numpy()
    a = permutation_feature_importance(trained_model, x, y, n_repeats=3, seed=42)
    b = permutation_feature_importance(trained_model, x, y, n_repeats=3, seed=42)
    assert a == b


# --- rank_features ---------------------------------------------------------- #
def test_rank_features_sorts_descending() -> None:
    imp = {"a": 0.1, "b": 0.5, "c": 0.3}
    ranked = rank_features(imp)
    assert [r.name for r in ranked] == ["b", "c", "a"]
    assert [r.rank for r in ranked] == [1, 2, 3]


def test_rank_features_top_k() -> None:
    imp = {"a": 0.1, "b": 0.5, "c": 0.3}
    ranked = rank_features(imp, top_k=2)
    assert len(ranked) == 2
    assert [r.name for r in ranked] == ["b", "c"]


def test_rank_features_share_sums_to_one_for_valid_names() -> None:
    # Use real feature names so index lookup succeeds.
    imp = {FEATURE_NAMES[0]: 0.5, FEATURE_NAMES[1]: 0.3, FEATURE_NAMES[2]: 0.2}
    ranked = rank_features(imp)
    assert abs(sum(r.share for r in ranked) - 1.0) < 1e-9


def test_rank_features_zero_total_returns_zero_share() -> None:
    imp = {FEATURE_NAMES[0]: 0.0, FEATURE_NAMES[1]: 0.0}
    ranked = rank_features(imp)
    assert all(r.share == 0.0 for r in ranked)
    assert all(r.importance == 0.0 for r in ranked)


# --- feature_importance_summary --------------------------------------------- #
def test_summary_gradient_only(trained_model: SurrogateModel) -> None:
    x = _make_batch(32)
    s = feature_importance_summary(trained_model, x, method="gradient")
    assert s.gradient
    assert not s.permutation
    assert s.gradient_ranked
    assert not s.permutation_ranked


def test_summary_both_methods(trained_model: SurrogateModel) -> None:
    x = _make_batch(32)
    sec, _ = trained_model(torch_from_np(x))
    y = sec.sum(dim=1).detach().cpu().numpy()
    s = feature_importance_summary(
        trained_model, x, y, method="both", n_repeats=2, top_k=5
    )
    assert s.gradient and s.permutation
    assert s.gradient_ranked and s.permutation_ranked
    assert 0.0 <= s.top_k_agreement <= 1.0


def test_summary_rejects_bad_method(trained_model: SurrogateModel) -> None:
    x = _make_batch(8)
    with pytest.raises(ValueError, match="method must be"):
        feature_importance_summary(trained_model, x, method="unknown")


def test_summary_permutation_requires_y(trained_model: SurrogateModel) -> None:
    x = _make_batch(8)
    with pytest.raises(ValueError, match="y is required"):
        feature_importance_summary(trained_model, x, method="permutation")


# --- fixture helper --------------------------------------------------------- #
def torch_from_np(arr: np.ndarray):
    import torch
    return torch.as_tensor(arr, dtype=torch.float32)
