from datetime import date

import pytest

from e2l_common.paths import DataPaths, DemoId


def test_demo_id_round_trip():
    d = DemoId.parse("20261004_bowlplate_007")
    assert d == DemoId(date(2026, 10, 4), "bowlplate", 7)
    assert str(d) == "20261004_bowlplate_007"
    with pytest.raises(ValueError):
        DemoId.parse("2026-10-04_bowlplate_7")


def test_layout(tmp_path):
    p = DataPaths(tmp_path)
    assert p.raw("s1", "x") == tmp_path / "raw/s1/x.mp4"
    assert p.manifest("run1") == tmp_path / "generated/run1/manifest.json"
    (tmp_path / "normalised").mkdir()
    for name in ("20261004_bowlplate_002.mp4", "20261004_bowlplate_001.mp4", "junk.mp4"):
        (tmp_path / "normalised" / name).touch()
    assert p.demo_ids() == ["20261004_bowlplate_001", "20261004_bowlplate_002"]
