"""Topology classification tests."""
from __future__ import annotations

import os
import tempfile

from tsugi_mend.topology import (
    classify_fallback,
    classify_from_hostnames,
    classify_from_nccl_topo,
    detect,
)


def test_classify_from_hostnames_two_racks():
    topo = classify_from_hostnames({
        0: "host-a", 1: "host-a", 2: "host-a", 3: "host-a",
        4: "host-b", 5: "host-b", 6: "host-b", 7: "host-b",
    })
    assert topo.detection_method == "hostname"
    assert topo.n_racks() == 2
    assert topo.is_cross_rack(0, 4)
    assert not topo.is_cross_rack(0, 1)
    assert not topo.is_cross_rack(4, 7)


def test_classify_from_hostnames_single_host():
    topo = classify_from_hostnames({0: "host-a", 1: "host-a"})
    assert topo.n_racks() == 1
    assert not topo.is_cross_rack(0, 1)


def test_classify_fallback_one_rack():
    topo = classify_fallback(4)
    assert topo.detection_method == "fallback"
    assert topo.n_racks() == 1
    assert not topo.is_cross_rack(0, 3)


def test_classify_from_nccl_topo_parses_cpu_grouping():
    xml = """<?xml version='1.0' encoding='UTF-8'?>
    <system>
      <cpu numaid="0">
        <gpu rank="0"/>
        <gpu rank="1"/>
        <gpu rank="2"/>
        <gpu rank="3"/>
      </cpu>
      <cpu numaid="1">
        <gpu rank="4"/>
        <gpu rank="5"/>
        <gpu rank="6"/>
        <gpu rank="7"/>
      </cpu>
    </system>"""
    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False) as f:
        f.write(xml)
        path = f.name
    try:
        topo = classify_from_nccl_topo(path, rank_count=8)
        assert topo is not None
        assert topo.detection_method == "nccl_topo_file"
        assert topo.n_racks() == 2
        assert topo.is_cross_rack(0, 4)
        assert not topo.is_cross_rack(0, 1)
    finally:
        os.unlink(path)


def test_classify_from_nccl_topo_missing_file_returns_none():
    topo = classify_from_nccl_topo("/nonexistent/topo.xml", rank_count=4)
    assert topo is None


def test_detect_prefers_nccl_topo_over_hostname():
    xml = """<?xml version='1.0' encoding='UTF-8'?>
    <system>
      <cpu numaid="0"><gpu rank="0"/></cpu>
      <cpu numaid="1"><gpu rank="1"/></cpu>
    </system>"""
    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False) as f:
        f.write(xml)
        path = f.name
    try:
        topo = detect(
            rank_count=2,
            nccl_topo_file=path,
            rank_to_hostname={0: "host-a", 1: "host-a"},  # would say 1 rack
        )
        assert topo.detection_method == "nccl_topo_file"
        assert topo.n_racks() == 2
    finally:
        os.unlink(path)


def test_detect_falls_through_to_hostname_then_fallback():
    # No NCCL file; hostname map present.
    topo = detect(
        rank_count=4,
        nccl_topo_file=None,
        rank_to_hostname={0: "host-a", 1: "host-a", 2: "host-b", 3: "host-b"},
    )
    assert topo.detection_method == "hostname"
    assert topo.n_racks() == 2

    # No NCCL file, no hostname map: fallback.
    topo = detect(rank_count=4, nccl_topo_file=None, rank_to_hostname=None)
    assert topo.detection_method == "fallback"
    assert topo.n_racks() == 1
