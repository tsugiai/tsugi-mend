"""MendConfig validation rule tests."""
from __future__ import annotations

import pytest

from tsugi_mend.config import MendConfig


def test_defaults_construct():
    c = MendConfig()
    assert c.quorum_min_learners == 4
    assert c.grace_window_ms == 2000
    assert c.token_weighted_merge is True
    assert c.sync_period_steps == 128
    assert c.momentum_sync_period_steps == 512
    assert c.async_tp_enabled is True
    assert c.failslow_zscore_threshold == 3.0
    assert c.failslow_window_steps == 50
    assert c.rack_aware is True
    assert c.sideband_addr == "tcp://0.0.0.0:51900"
    assert c.sideband_peers == ()
    assert c.diagnostics_dir is None


def test_quorum_must_be_positive():
    with pytest.raises(ValueError, match="quorum_min_learners"):
        MendConfig(quorum_min_learners=0)


def test_grace_window_nonnegative():
    MendConfig(grace_window_ms=0)  # zero is allowed (immediate close after K-th)
    with pytest.raises(ValueError, match="grace_window_ms"):
        MendConfig(grace_window_ms=-1)


def test_outer_momentum_range():
    MendConfig(outer_optimizer_momentum=0.0)
    MendConfig(outer_optimizer_momentum=0.99)
    with pytest.raises(ValueError, match="outer_optimizer_momentum"):
        MendConfig(outer_optimizer_momentum=1.0)
    with pytest.raises(ValueError, match="outer_optimizer_momentum"):
        MendConfig(outer_optimizer_momentum=-0.01)


def test_des_loc_M_ge_N():
    """DES-LOC requires momentum-sync period >= param-sync period."""
    MendConfig(sync_period_steps=128, momentum_sync_period_steps=128)
    MendConfig(sync_period_steps=128, momentum_sync_period_steps=512)
    with pytest.raises(ValueError, match="DES-LOC requires M >= N"):
        MendConfig(sync_period_steps=128, momentum_sync_period_steps=64)


def test_failslow_validation():
    with pytest.raises(ValueError, match="failslow_window_steps"):
        MendConfig(failslow_window_steps=1)
    with pytest.raises(ValueError, match="failslow_zscore_threshold"):
        MendConfig(failslow_zscore_threshold=0)
    with pytest.raises(ValueError, match="failslow_min_samples"):
        MendConfig(failslow_min_samples=1)
    with pytest.raises(ValueError, match="cannot exceed"):
        MendConfig(failslow_window_steps=10, failslow_min_samples=20)


def test_sideband_addr_validation():
    with pytest.raises(ValueError, match="sideband_addr"):
        MendConfig(sideband_addr="udp://0.0.0.0:51900")
    with pytest.raises(ValueError, match="sideband_peers"):
        MendConfig(sideband_peers=("udp://peer:1234",))


def test_sideband_heartbeat_positive():
    with pytest.raises(ValueError, match="sideband_heartbeat_ms"):
        MendConfig(sideband_heartbeat_ms=0)


def test_sideband_inbound_limits_validation():
    with pytest.raises(ValueError, match="sideband_inbound_read_timeout_s"):
        MendConfig(sideband_inbound_read_timeout_s=0)
    with pytest.raises(ValueError, match="sideband_max_inbound_connections"):
        MendConfig(sideband_max_inbound_connections=0)


def test_sideband_tls_requires_ca_file():
    with pytest.raises(ValueError, match="sideband_tls_ca_file"):
        MendConfig(
            sideband_tls=True,
            sideband_tls_certfile="server.crt",
            sideband_tls_keyfile="server.key",
        )


def test_auto_tune_runtime_defaults_off():
    c = MendConfig()
    assert c.auto_tune_runtime is False
    assert c.auto_tune_runtime_window_steps == 50
    assert c.auto_tune_runtime_min_samples == 10
    assert c.auto_tune_zscore_min == 2.0
    assert c.auto_tune_zscore_max == 8.0
    assert c.auto_tune_grace_window_min_ms == 0
    assert c.auto_tune_grace_window_max_ms == 10_000
    assert c.auto_tune_cov_gain == 4.0
    assert c.auto_tune_grace_gain == 1.0


def test_auto_tune_runtime_validation():
    with pytest.raises(ValueError, match="auto_tune_runtime_window_steps"):
        MendConfig(auto_tune_runtime_window_steps=1)
    with pytest.raises(ValueError, match="auto_tune_runtime_min_samples"):
        MendConfig(auto_tune_runtime_min_samples=1)
    with pytest.raises(ValueError, match="cannot exceed"):
        MendConfig(
            auto_tune_runtime_window_steps=10,
            auto_tune_runtime_min_samples=20,
        )
    with pytest.raises(ValueError, match="auto_tune_zscore_min"):
        MendConfig(auto_tune_zscore_min=0)
    with pytest.raises(ValueError, match="auto_tune_zscore_max"):
        MendConfig(auto_tune_zscore_min=5.0, auto_tune_zscore_max=3.0)
    with pytest.raises(ValueError, match="auto_tune_grace_window_min_ms"):
        MendConfig(auto_tune_grace_window_min_ms=-1)
    with pytest.raises(ValueError, match="auto_tune_grace_window_max_ms"):
        MendConfig(
            auto_tune_grace_window_min_ms=5000,
            auto_tune_grace_window_max_ms=1000,
        )
    with pytest.raises(ValueError, match="auto_tune_cov_gain"):
        MendConfig(auto_tune_cov_gain=-1.0)
    with pytest.raises(ValueError, match="auto_tune_grace_gain"):
        MendConfig(auto_tune_grace_gain=-1.0)
