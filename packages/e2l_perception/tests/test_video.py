import shutil
import subprocess

import numpy as np
import pytest

from e2l_perception.video import VideoReader, normalise

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def test_normalise_to_constant_rate_and_read(tmp_path):
    src = tmp_path / "raw.mp4"
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i",
         "testsrc=size=160x120:rate=24:duration=1", str(src)],
        check=True,
    )  # fmt: skip
    dst = normalise(src, tmp_path / "norm" / "demo.mp4", fps=30)
    reader = VideoReader(dst)
    assert reader.fps == 30 and reader.size == (160, 120)
    frames = list(reader)
    assert len(frames) == 30
    t = np.array([ti for _, ti in frames])
    np.testing.assert_allclose(t, np.arange(30) / 30)
    assert frames[0][0].shape == (120, 160, 3)


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        VideoReader(tmp_path / "nope.mp4")
