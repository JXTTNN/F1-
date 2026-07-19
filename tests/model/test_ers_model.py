"""Tests for f1opt.model.ers_model (Iter-5)."""

from __future__ import annotations

import pytest

from f1opt.model.ers_model import (
    ERS_TRACK_PROFILES,
    ERSDeploymentModel,
    ERSDeploymentZone,
    ERSTrackProfile,
    get_ers_profile,
)


# --------------------------------------------------------------------------- #
# ERSDeploymentZone
# --------------------------------------------------------------------------- #
class TestERSDeploymentZone:
    def test_from_segment_computes_length(self) -> None:
        z = ERSDeploymentZone.from_segment(100.0, 1100.0)
        assert z.length_m == 1000.0
        assert z.start_m == 100.0
        assert z.end_m == 1100.0

    def test_drag_limited_when_long(self) -> None:
        z = ERSDeploymentZone.from_segment(0.0, 800.0)
        assert z.is_drag_limited is True

    def test_not_drag_limited_when_short(self) -> None:
        z = ERSDeploymentZone.from_segment(0.0, 300.0)
        assert z.is_drag_limited is False

    def test_default_priority_by_length(self) -> None:
        z_short = ERSDeploymentZone.from_segment(0.0, 500.0)
        z_long = ERSDeploymentZone.from_segment(0.0, 1500.0)
        assert z_long.priority_weight > z_short.priority_weight

    def test_negative_length_clamped(self) -> None:
        z = ERSDeploymentZone.from_segment(500.0, 100.0)
        assert z.length_m == 0.0


# --------------------------------------------------------------------------- #
# ERSTrackProfile
# --------------------------------------------------------------------------- #
class TestERSTrackProfile:
    def test_total_deploy_length(self) -> None:
        p = get_ers_profile("monza")
        assert p.total_deploy_length_m() > 0.0

    def test_total_harvest_mj(self) -> None:
        p = get_ers_profile("monza")
        assert p.total_harvest_mj() > 0.0

    def test_longest_drag_limited_zone(self) -> None:
        p = get_ers_profile("spa")
        z = p.longest_drag_limited_zone()
        assert z is not None
        assert z.is_drag_limited

    def test_longest_drag_limited_zone_none_when_no_dl(self) -> None:
        # Construct profile with only short zones
        p = ERSTrackProfile(
            track_id="test", lap_length_m=2000.0,
            deployment_zones=(ERSDeploymentZone.from_segment(0.0, 200.0),),
            harvest_zones=((150.0, 200.0, 0.3),),
        )
        assert p.longest_drag_limited_zone() is None

    def test_get_profile_unknown_returns_default(self) -> None:
        p = get_ers_profile("nonexistent_track")
        assert p.track_id == "unknown"
        assert len(p.deployment_zones) > 0


# --------------------------------------------------------------------------- #
# 12 track profiles data integrity
# --------------------------------------------------------------------------- #
class TestTrackProfileDataIntegrity:
    def test_all_12_profiles_present(self) -> None:
        expected = {"monza", "spa", "jeddah", "bahrain", "silverstone",
                    "monaco", "suzuka", "melbourne", "yas_marina",
                    "shanghai", "austin", "interlagos"}
        assert expected.issubset(set(ERS_TRACK_PROFILES.keys()))

    def test_each_profile_has_at_least_one_zone(self) -> None:
        for tid, p in ERS_TRACK_PROFILES.items():
            assert len(p.deployment_zones) >= 1, f"{tid} missing deploy zones"
            assert len(p.harvest_zones) >= 1, f"{tid} missing harvest zones"

    def test_each_profile_has_drag_limited_zone(self) -> None:
        """All real F1 tracks have at least one long straight, EXCEPT Monaco
        (street circuit with no long straights)."""
        for tid, p in ERS_TRACK_PROFILES.items():
            if tid == "monaco":
                # Monaco has no drag-limited zone — physically correct.
                assert p.longest_drag_limited_zone() is None
                continue
            assert p.longest_drag_limited_zone() is not None, f"{tid} no DL zone"


# --------------------------------------------------------------------------- #
# ERSDeploymentModel basics
# --------------------------------------------------------------------------- #
class TestERSDeploymentModelBasics:
    def test_invalid_mode_falls_back_to_balanced(self) -> None:
        m = ERSDeploymentModel(track_id="monza", mode="invalid")
        assert m.mode == "balanced"

    def test_initial_soc_clamped(self) -> None:
        m = ERSDeploymentModel(track_id="monza", initial_soc=2.0)
        assert m.initial_soc == 1.0
        m2 = ERSDeploymentModel(track_id="monza", initial_soc=-0.5)
        assert m2.initial_soc == 0.0

    def test_simulate_lap_returns_required_keys(self) -> None:
        m = ERSDeploymentModel(track_id="monza")
        r = m.simulate_lap()
        required = {"track_id", "mode", "deploy_mj", "harvest_mj", "net_mj",
                    "soc_before", "soc_after", "lap_gain_s",
                    "lap_drag_cost_s", "net_lap_gain_s", "zone_breakdown"}
        assert required.issubset(r.keys())

    def test_zone_breakdown_has_per_zone_data(self) -> None:
        m = ERSDeploymentModel(track_id="monza")
        r = m.simulate_lap()
        for z in r["zone_breakdown"]:
            assert "start_m" in z and "end_m" in z and "mj_deployed" in z
            assert z["mj_deployed"] >= 0.0


# --------------------------------------------------------------------------- #
# Physics: deploy/harvest/SoC
# --------------------------------------------------------------------------- #
class TestERSPhysics:
    def test_soc_stays_in_unit_interval(self) -> None:
        """SoC must never go below 0 or above 1 across a long stint."""
        m = ERSDeploymentModel(track_id="monza", mode="attack", initial_soc=0.5)
        for _ in range(50):
            m.simulate_lap()
            assert 0.0 <= m.soc <= 1.0

    def test_deploy_cannot_exceed_soc_available(self) -> None:
        """If SoC=0, deploy must be 0 (no energy to spend)."""
        m = ERSDeploymentModel(track_id="monza", mode="attack", initial_soc=0.0)
        r = m.simulate_lap()
        assert r["deploy_mj"] == 0.0

    def test_attack_deploys_more_than_conserve(self) -> None:
        m_atk = ERSDeploymentModel(track_id="monza", mode="attack", initial_soc=0.9)
        m_con = ERSDeploymentModel(track_id="monza", mode="conserve", initial_soc=0.9)
        r_atk = m_atk.simulate_lap()
        r_con = m_con.simulate_lap()
        assert r_atk["deploy_mj"] > r_con["deploy_mj"]

    def test_attack_net_gain_higher_than_conserve(self) -> None:
        m_atk = ERSDeploymentModel(track_id="monza", mode="attack", initial_soc=0.9)
        m_con = ERSDeploymentModel(track_id="monza", mode="conserve", initial_soc=0.9)
        r_atk = m_atk.simulate_lap()
        r_con = m_con.simulate_lap()
        assert r_atk["net_lap_gain_s"] > r_con["net_lap_gain_s"]

    def test_attack_drains_soc_more_than_conserve(self) -> None:
        m_atk = ERSDeploymentModel(track_id="monza", mode="attack", initial_soc=0.9)
        m_con = ERSDeploymentModel(track_id="monza", mode="conserve", initial_soc=0.9)
        r_atk = m_atk.simulate_lap()
        r_con = m_con.simulate_lap()
        assert r_atk["soc_after"] < r_con["soc_after"]

    def test_harvest_charges_battery(self) -> None:
        """Low SoC + conserve → SoC should increase over time."""
        m = ERSDeploymentModel(track_id="monza", mode="conserve", initial_soc=0.1)
        soc_before = m.soc
        m.simulate_lap()
        assert m.soc > soc_before

    def test_deploy_respects_fia_cap(self) -> None:
        """Deploy cannot exceed FIA 4 MJ cap even with full SoC + attack mode."""
        m = ERSDeploymentModel(track_id="monza", mode="attack", initial_soc=1.0)
        r = m.simulate_lap()
        # 4 MJ cap × 1.3 attack factor = 5.2 MJ; but also capped by SoC (4 MJ).
        # So max deploy = min(5.2, 4.0) = 4.0 MJ.
        assert r["deploy_mj"] <= 4.0 + 1e-6

    def test_harvest_respects_fia_cap(self) -> None:
        """Harvest cannot exceed FIA 2.5 MJ cap."""
        # Use a profile with very high harvest request
        big_profile = ERSTrackProfile(
            track_id="test", lap_length_m=5000.0,
            deployment_zones=(ERSDeploymentZone.from_segment(0.0, 700.0),),
            harvest_zones=((0.0, 100.0, 5.0), (200.0, 300.0, 5.0)),  # 10 MJ requested
        )
        # Manually use profile
        m = ERSDeploymentModel(track_id="unknown", mode="balanced", initial_soc=0.5)
        m.profile = big_profile
        r = m.simulate_lap()
        assert r["harvest_mj"] <= 2.5 + 1e-6


# --------------------------------------------------------------------------- #
# Stint simulation
# --------------------------------------------------------------------------- #
class TestStintSimulation:
    def test_stint_returns_n_laps(self) -> None:
        m = ERSDeploymentModel(track_id="monza")
        laps = m.simulate_stint(20)
        assert len(laps) == 20
        assert [lp["lap"] for lp in laps] == list(range(1, 21))

    def test_stint_resets_soc(self) -> None:
        """simulate_stint always starts from initial_soc."""
        m = ERSDeploymentModel(track_id="monza", initial_soc=0.5)
        # Modify internal SoC
        m._soc = 0.99
        laps = m.simulate_stint(5)
        assert laps[0]["soc_before"] == pytest.approx(0.5, abs=0.01)

    def test_stint_soc_stays_in_range(self) -> None:
        m = ERSDeploymentModel(track_id="monza", mode="attack", initial_soc=0.5)
        for lp in m.simulate_stint(50):
            assert 0.0 <= lp["soc_after"] <= 1.0

    def test_reset(self) -> None:
        m = ERSDeploymentModel(track_id="monza", initial_soc=0.5)
        m.simulate_lap()
        m.reset()
        assert m.soc == 0.5


# --------------------------------------------------------------------------- #
# Mode recommendation
# --------------------------------------------------------------------------- #
class TestModeRecommendation:
    def test_high_soc_recommends_attack(self) -> None:
        m = ERSDeploymentModel(track_id="monza", initial_soc=0.9)
        assert m.recommend_mode(target_soc=0.5) == "attack"

    def test_low_soc_recommends_conserve(self) -> None:
        m = ERSDeploymentModel(track_id="monza", initial_soc=0.1)
        assert m.recommend_mode(target_soc=0.5) == "conserve"

    def test_mid_soc_recommends_balanced(self) -> None:
        m = ERSDeploymentModel(track_id="monza", initial_soc=0.5)
        assert m.recommend_mode(target_soc=0.5) == "balanced"


# --------------------------------------------------------------------------- #
# Robustness
# --------------------------------------------------------------------------- #
class TestRobustness:
    def test_unknown_track_uses_default_profile(self) -> None:
        m = ERSDeploymentModel(track_id="nonexistent", mode="balanced")
        r = m.simulate_lap()
        assert r["track_id"] == "nonexistent"
        assert len(r["zone_breakdown"]) > 0

    def test_zero_lap_stint(self) -> None:
        m = ERSDeploymentModel(track_id="monza")
        assert m.simulate_stint(0) == []

    def test_soc_at_zero_still_runs(self) -> None:
        m = ERSDeploymentModel(track_id="monza", initial_soc=0.0)
        r = m.simulate_lap()
        # Deploy is 0 but harvest charges the battery
        assert r["deploy_mj"] == 0.0
        assert r["harvest_mj"] > 0.0
        assert r["soc_after"] >= 0.0
